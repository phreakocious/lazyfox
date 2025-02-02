#!/usr/bin/env python3

########################
# lazyfox.py          #
# phreakocious, 2025 #
#####################

# LazyFox is a parallelized implementation of Fox -
# a community detection algorithm for undirected graphs with support for overlapping communities
#
# most similar algorithms can only assign a node to a single community, where this one can do multiple
#
# algorithm paper  -- https://peerj.com/articles/cs-1291.pdf
# adapted from c++ -- https://github.com/timgarrels/LazyFox
#
# this version was created to get an understanding of the algorithm and experiment with improvements.
# it's also been useful for testing the python 3.13 non-GIL interpreter with a highly threaded workload.
# the original c++ version is considerably faster.

# TODO: test cluster loader and post-processing
# TODO: allow other file types
# TODO: pass graph into fox to make it more portable
# TODO: dump node-keyed output at end to accompany the community-keyed output


import os
import sys
import time
import json
import argparse
import sysconfig
import networkx as nx

from tqdm import tqdm
from enum import Enum
from itertools import islice
from datetime import datetime
from threading import Thread, Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def debug(vars):  # drop in wherever for a REPL with debug(locals())
    import code
    import readline
    import rlcompleter

    from rich import pretty, inspect

    pp = inspect
    pretty.install()

    if "libedit" in readline.__doc__:  # handle macos
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    _all_vars = globals()
    _all_vars.update(vars)
    _all_vars.update(locals())
    readline.set_completer(rlcompleter.Completer(_all_vars).complete)

    code.InteractiveConsole(_all_vars).interact()


def intersection_size(list1, list2):
    return len(set(list1).intersection(set(list2)))


class ChangeType(Enum):
    STAY = 1
    COPY = 2
    LEAVE = 3
    TRANSFER = 4
    NONE = 5


class Change:
    def __init__(self, node_id=-1, source_community=-1, target_community=-1):
        self.node_id = node_id
        self.source_community = source_community
        self.target_community = target_community

    def get_type(self):
        if self.source_community == -1 and self.target_community == -1:
            return ChangeType.STAY
        if self.source_community == -1 and self.target_community != -1:
            return ChangeType.COPY
        if self.source_community != -1 and self.target_community == -1:
            return ChangeType.LEAVE
        if self.source_community != -1 and self.target_community != -1:
            return ChangeType.TRANSFER
        return ChangeType.NONE

    def __str__(self):
        return (
            f"{', '.join([f'{k}: {v}' for k, v in self.__dict__.items()])}, type={self.get_type()}"
        )


class ChangeCounter:
    def __init__(self):
        self.stays = 0
        self.copies = 0
        self.leaves = 0
        self.transfers = 0

    def add(self, change):
        change_type = change.get_type()
        if change_type == ChangeType.STAY:
            self.stays += 1
        elif change_type == ChangeType.COPY:
            self.copies += 1
        elif change_type == ChangeType.LEAVE:
            self.leaves += 1
        elif change_type == ChangeType.TRANSFER:
            self.transfers += 1
        # If ChangeType is NONE, we do nothing

    def __str__(self):
        return ", ".join([f"{k}: {v}" for k, v in self.__dict__.items()])


# TODO: test this or eliminate the feature.................
class ClusteringLoader:
    def __init__(self, filelocation, num_nodes):
        self.filelocation = filelocation
        self.num_nodes = num_nodes
        self.nodeToCluster = []
        self.maximumCommunityId = -1
        self.init()

    def init(self):
        if not os.path.isfile(self.filelocation):
            raise ValueError("Clustering file does not exist")

        self.nodeToCluster = [
            [] for _ in range(self.num_nodes)
        ]  # Initialize empty lists for each node
        communityId = 0

        with open(self.filelocation, "r") as infile:
            for line in infile:
                if line.startswith("#"):  # Skip comment lines
                    continue

                node_ids = list(map(int, line.split()))  # Read all node IDs in the line
                # Assume first item in the line is the community ID and others are node IDs
                for nodeId in node_ids[1:]:
                    self.nodeToCluster[nodeId].append(communityId)

                communityId += 1

        # For each node not mentioned in pre-clustering, create a single node community
        for nodeId in range(self.num_nodes):
            if not self.nodeToCluster[nodeId]:
                self.nodeToCluster[nodeId].append(communityId)
                communityId += 1

        self.maximumCommunityId = communityId - 1


class Fox:
    def __init__(
        self,
        input_file_path,
        output_dir,
        threshold=0.01,
        queue_size=1,
        thread_count=1,
        dump_results=True,
        clustering_path=None,
    ):
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.queue_size = queue_size
        self.thread_count = thread_count
        self.dump_results = dump_results

        self.iteration_count = 1
        self.cc = 0
        self.global_wcc = 0
        self.wcc_diff = 0

        self.mutex = Lock()

        print("creating output directories")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"FOX_{os.path.basename(input_file_path)}_{timestamp}"
        self.iteration_dump_dir = os.path.join(output_dir, folder_name, "iterations")
        os.makedirs(self.iteration_dump_dir, exist_ok=True)

        print("loading input file")
        self.G = nx.read_edgelist(self.input_file_path, nodetype=int)

        print("renumbering nodes")
        self.G = nx.convert_node_labels_to_integers(self.G, label_attribute="old_id")

        print("building the adjacency lists")
        self.adjacency_list = {node: list(sorted(self.G.neighbors(node))) for node in self.G.nodes}

        self.number_nodes = self.G.number_of_nodes()
        print(f"graph loaded - {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        self.hashmap_nc = defaultdict(list)
        self.hashmap_cwcc = {}
        self.hashmap_clookup = defaultdict(dict)
        self.number_neighbors = []
        self.cc_per_node = []

        print("beginning initialization")
        print("counting neighbors")
        self.count_neighbors_per_node()

        print("calculating CC Per node")
        self.calculate_cc_per_node()

        print("ordering nodes")
        self.order_nodes()

        print("computing CC global: ", end="")
        self.cc = self.calculate_global_cc()
        print(self.cc)

        if clustering_path:
            print("loading external clustering")
            self.load_clustering(clustering_path)
        else:
            print("initial clustering")
            self.initialize_cluster()

        print(f"initial clustering produced {self.maximum_community_id + 1} communities")
        self.initialize_cluster_maps()

        print("global WCC: ", end="")
        self.global_wcc = sum(self.hashmap_cwcc.values())
        print(self.global_wcc)

        print("cleaning broken communities")
        self.remove_all_single_node_communities()
        print(f"{len(self.hashmap_cwcc)} communities remaining")

        print("initialization done")

    def initialize_cluster(self):
        max_community_id = -1

        for node_id in self.node_ids:
            if self.hashmap_nc[node_id]:
                continue
            max_community_id += 1
            self.hashmap_nc[node_id].append(max_community_id)

            for neighbor_id in self.adjacency_list[node_id]:
                if not self.hashmap_nc[neighbor_id]:
                    self.hashmap_nc[neighbor_id].append(max_community_id)
        self.maximum_community_id = max_community_id

    def initialize_cluster_maps(self):
        """
        Initializes the following lookups:
        1. From community to node and from that node to its edge count within the community.
        2. From community to its WCC score.
        """

        # initialize the community to node and edge count mapping
        for node_id in self.node_ids:
            # Nodes have exactly one community at this stage
            community = self.hashmap_nc[node_id][0]
            cmi = self.hashmap_clookup.get(community, {})
            cmi[node_id] = 0  # Initialize the edge count for the node
            self.hashmap_clookup[community] = cmi

        # update the edge count for each node in each community
        for community_id in range(self.maximum_community_id + 1):
            cmi = self.hashmap_clookup.get(community_id, {})
            nodes_in_community = list(cmi.keys())  # All node ids in the current community

            # calculate intersection size for each node in the community
            for node_id, wcc_in_community in cmi.items():
                # Update the edge count for the node
                cmi[node_id] = float(
                    intersection_size(nodes_in_community, self.adjacency_list[node_id])
                )

            self.hashmap_clookup[community_id] = cmi

        # calculate the WCC score for each community
        for community_id in range(self.maximum_community_id + 1):
            community = self.hashmap_clookup.get(community_id, {})
            self.hashmap_cwcc[community_id] = self.calculate_wcc_dach_community(community)

        print("finished the hashmap population.")

    # NOTE: C++ version had its own threaded implementation that may be faster on massive graphs..
    def calculate_cc_per_node(self):
        cc = nx.clustering(self.G)
        self.cc_per_node = [cc[i] for i in range(self.number_nodes)]

    def calculate_global_cc(self):
        return sum(self.cc_per_node) / float(self.number_nodes)

    def count_neighbors_per_node(self):
        self.number_neighbors = [len(neighbors) for neighbors in self.adjacency_list.values()]

    def order_nodes(self):
        self.node_ids = list(range(self.number_nodes))

        # stable sort the nodes by number of neighbors in descending order
        self.node_ids.sort(key=lambda a: self.number_neighbors[a], reverse=True)

        # stable sort the nodes again by clustering coefficient in descending order
        self.node_ids.sort(key=lambda a: self.cc_per_node[a], reverse=True)

    def calculate_wcc_dach(self, node_id, community):
        if len(community) <= 1:
            return 0

        node_degree = self.number_neighbors[node_id]

        if node_degree <= 1:  # nodes with one edge or less cannot form triangles
            return 0

        node_degree_to_community = community[node_id]
        node_degree_to_graph_without_community = float(node_degree) - node_degree_to_community
        edges_in_community = sum(community.values()) / 2.0

        possible_edges_in_community = float(len(community) * (len(community) - 1) / 2.0)
        community_density = edges_in_community / possible_edges_in_community

        expected_triangles_with_community = (
            node_degree_to_community * (node_degree_to_community - 1) / 2.0
        ) * community_density

        expected_triangles_with_graph = (node_degree * (node_degree - 1) / 2.0) * self.cc

        # WCC Dach score
        wcc_dach = (
            (expected_triangles_with_community / expected_triangles_with_graph)
            * node_degree
            / (len(community) - 1 + node_degree_to_graph_without_community)
        )

        return wcc_dach

    def calculate_wcc_dach_community(self, community):
        acc = 0
        for node_id, _ in community.items():
            acc += self.calculate_wcc_dach(node_id, community)
        return acc

    def remove_single_node_community(self, community_id):
        if len(self.hashmap_clookup.get(community_id, {})) > 1:
            raise ValueError("Community contains more than 1 node!")

        last_node_id = next(iter(self.hashmap_clookup[community_id]))

        del self.hashmap_clookup[community_id]
        del self.hashmap_cwcc[community_id]

        if last_node_id in self.hashmap_nc:
            self.hashmap_nc[last_node_id] = [
                community
                for community in self.hashmap_nc[last_node_id]
                if community != community_id
            ]

    def remove_all_single_node_communities(self):
        for community_id in range(self.maximum_community_id + 1):
            if (
                community_id in self.hashmap_clookup
                and len(self.hashmap_clookup[community_id]) <= 1
            ):
                self.remove_single_node_community(community_id)

    def get_altered_community_leave(self, node_id, community_id):
        if community_id not in self.hashmap_clookup:
            raise ValueError("Community id should not exist")
        altered_community = self.hashmap_clookup[community_id].copy()
        for neighbor in self.adjacency_list[node_id]:
            if community_id in self.hashmap_nc[neighbor]:
                altered_community[neighbor] = altered_community.get(neighbor, 0) - 1
        del altered_community[node_id]
        return altered_community

    def get_altered_community_join(self, node_id, community_id):
        altered_community = self.hashmap_clookup[community_id].copy()
        altered_community[node_id] = 0
        for neighbor in self.adjacency_list[node_id]:
            if community_id in self.hashmap_nc[neighbor]:
                altered_community[neighbor] = altered_community.get(neighbor, 0) + 1
                altered_community[node_id] = altered_community.get(node_id, 0) + 1
        return altered_community

    def calculate_delta_l(self, node_id, community_id):
        altered_community = self.get_altered_community_leave(node_id, community_id)
        wcc_dach_altered_community = self.calculate_wcc_dach_community(altered_community)
        delta_l = wcc_dach_altered_community - self.hashmap_cwcc[community_id]
        return delta_l

    def calculate_delta_j(self, node_id, community_id):
        altered_community = self.get_altered_community_join(node_id, community_id)
        wcc_dach_altered_community = self.calculate_wcc_dach_community(altered_community)
        delta_j = wcc_dach_altered_community - self.hashmap_cwcc[community_id]
        return delta_j

    def community_to_leave(self, node_id):
        best_move_delta_l = 0
        best_cid = -1
        for community_id in self.hashmap_nc[node_id]:
            delta_l = self.calculate_delta_l(node_id, community_id)
            if delta_l > best_move_delta_l:
                best_move_delta_l = delta_l
                best_cid = community_id
        return best_cid

    def community_to_join(self, node_id):
        best_move_delta_j = 0
        best_cid = -1
        relevant_communities = set()
        for neighbor in self.adjacency_list[node_id]:
            for community_id in self.hashmap_nc[neighbor]:
                relevant_communities.add(community_id)
        for community_id in self.hashmap_nc[node_id]:
            relevant_communities.discard(community_id)
        for community_id in relevant_communities:
            delta_j = self.calculate_delta_j(node_id, community_id)
            if delta_j > best_move_delta_j:
                best_move_delta_j = delta_j
                best_cid = community_id
        return best_cid

    def decide(self, node_id):
        change = Change(node_id, -1, -1)
        change.source_community = self.community_to_leave(node_id)
        change.target_community = self.community_to_join(node_id)
        return change

    def apply(self, change):
        if change.get_type() == ChangeType.TRANSFER or change.get_type() == ChangeType.COPY:
            altered_community = self.get_altered_community_join(
                change.node_id, change.target_community
            )
            new_wcc = self.calculate_wcc_dach_community(altered_community)
            self.wcc_diff += new_wcc - self.hashmap_cwcc[change.target_community]
            self.hashmap_clookup[change.target_community] = altered_community
            self.hashmap_cwcc[change.target_community] = new_wcc
            self.hashmap_nc[change.node_id].append(change.target_community)

        if change.get_type() == ChangeType.TRANSFER or change.get_type() == ChangeType.LEAVE:
            altered_community = self.get_altered_community_leave(
                change.node_id, change.source_community
            )
            new_wcc = self.calculate_wcc_dach_community(altered_community)
            self.wcc_diff += new_wcc - self.hashmap_cwcc[change.source_community]
            self.hashmap_clookup[change.source_community] = altered_community
            self.hashmap_cwcc[change.source_community] = new_wcc
            self.hashmap_nc[change.node_id].remove(change.source_community)

    def dump_results_to_file(self, change_counter, duration):
        with open(
            os.path.join(self.iteration_dump_dir, f"{self.iteration_count}clusters.txt"),
            "w",
        ) as output:
            for cluster_id in range(self.maximum_community_id + 1):
                if cluster_id not in self.hashmap_clookup:
                    continue
                for k, v in self.hashmap_clookup[cluster_id].items():
                    output.write(f"{k}\t")
                output.write("\n")

        with open(
            os.path.join(self.iteration_dump_dir, f"{self.iteration_count}.json"), "w"
        ) as output:
            cwcc_data = {str(k): v for k, v in self.hashmap_cwcc.items()}
            result_data = {
                "change_counter": change_counter,
                "iteration": self.iteration_count,
                "runtime": duration,
                "wcc_lookup": cwcc_data,
            }
            output.write(json.dumps(result_data, indent=4, default=vars))

    def p(self, msg):
        print(f"iteration {self.iteration_count} - {msg}")

    def process_node(self, node_position, change_counter):
        if node_position >= self.number_nodes:
            return None  # Skip out-of-bound nodes

        node_id = self.node_ids[node_position]
        change = self.decide(node_id)
        change_counter.add(change)
        return change

    def process_nodes(self, chunk_start, chunk_end, change_counter, pbars):
        with self.mutex:
            pbar = next((p for p in pbars if p.ready))  # grab the next available progress bar
            pbar.ready = False  # ensure nobody steps on it

        pbar.reset()  # reset it if it was previously used
        changes = []
        for offset in range(chunk_start, chunk_end):
            if change := self.process_node(offset, change_counter):
                changes.append(change)
            pbar.update(1)
        pbar.refresh()
        pbar.ready = True  # free up progress bar for next thread
        return changes

    def process_stuff3(self, change_counter):
        i = f"iteration {self.iteration_count}"
        chunk_offsets = list(islice(range(self.number_nodes), 0, None, self.queue_size))
        changes = []
        futures = []

        pbars = [
            tqdm(position=t, desc=f"{i} - thread {t}", total=self.queue_size, leave=False)
            for t in range(1, self.thread_count + 1)
        ]
        for p in pbars:
            p.ready = True

        pbar = tqdm(
            desc=f"{i} - processing", total=self.number_nodes, position=self.thread_count + 1
        )
        pbars.append(pbar)
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            for chunk_start in chunk_offsets:
                chunk_end = chunk_start + self.queue_size
                futures.append(
                    executor.submit(
                        self.process_nodes, chunk_start, chunk_end, change_counter, pbars
                    )
                )

            for future in as_completed(futures):
                try:
                    result = future.result()
                    pbar.update(len(result))
                    changes.extend(result)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"broke - {e}")

        for p in pbars:
            p.close()
        for change in tqdm(changes, desc=f"{i} - applying changes"):
            self.apply(change)

    def run(self):
        print("starting the main loop")
        print(f"{self.threshold} is the threshold")

        while True:
            change_counter = ChangeCounter()
            self.wcc_diff = 0.0
            start_time = time.time()

            self.p("start")
            self.process_stuff3(change_counter)
            self.remove_all_single_node_communities()
            self.p("done")

            self.p(change_counter)

            relative_change = self.wcc_diff / self.global_wcc
            self.global_wcc += self.wcc_diff
            self.p(f"relative change {relative_change:.6f}")

            self.p(f"{len(self.hashmap_cwcc)} communities remaining")

            end_time = time.time()
            elapsed_seconds = end_time - start_time
            self.p(f"epoch took {elapsed_seconds:.2f}s")

            done = (change_counter.stays == self.number_nodes) or (
                relative_change < self.threshold
            )

            if self.dump_results or done:
                self.p("dumping results")
                self.dump_results_to_file(change_counter, elapsed_seconds)
                self.p("results dumped successfully")

            print("")

            if done:
                break
            self.iteration_count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Run the LazyFox algorithm\n\n"
        "Overlapping community detection for very large graph datasets with billions of edges\n"
        "by optimizing a WCC estimation."
    )

    parser.add_argument(
        "--input-graph",
        required=True,
        type=str,
        help="File containing the graph dataset as an edge list",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory, a subdirectory will be created for each run",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=1,
        help="The degree of parallel processing (default=1)",
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        default=1,
        help="How many threads to use. Should be below or equal to queue_size (default=1)",
    )
    parser.add_argument(
        "--wcc-threshold",
        type=float,
        default=0.01,
        help="Threshold in wcc-change to stop processing (default=0.01)",
    )
    parser.add_argument(
        "--disable-dumping",
        action="store_true",
        help="LazyFox will only save the clustering results of the final iteration to disk, not intermediate results",
    )
    parser.add_argument(
        "--pre-clustering",
        type=str,
        help="Loads external node clustering, replacing initial clustering algorithm",
    )
    parser.add_argument(
        "--post-processing",
        type=str,
        help="Script will be called after LazyFox clustering, with the run-subdirectory as argument",
    )

    args = parser.parse_args()

    print("starting lazyfox.py")
    gil_status = sysconfig.get_config_var("Py_GIL_DISABLED")
    if gil_status is None:
        print("python GIL cannot be disabled")
    elif gil_status == 0:
        print("python GIL is active")
    elif gil_status == 1:
        print("python GIL is disabled")

    print(f"loading input graph from {args.input_graph}")
    print(f"writing output to {args.output_dir}")
    print(f"WCC threshold set to {args.wcc_threshold}")
    print(f"thread count set to {args.thread_count}")
    print(f"queue size set to {args.queue_size}")
    print(f"dumping set to {not args.disable_dumping}")
    if args.pre_clustering:
        print(f"loading pre-clustering from {args.pre_clustering}")
    if args.post_processing:
        print(f"post-processing script enabled: {args.post_processing}")

    start_time = time.time()
    fox = Fox(
        args.input_graph,
        args.output_dir,
        args.wcc_threshold,
        args.queue_size,
        args.thread_count,
        not args.disable_dumping,
        args.pre_clustering,
    )
    fox.run()

    elapsed_time = time.time() - start_time
    print(f"finished computation. elapsed time: {elapsed_time:.2f}s")

    if args.post_processing:
        print("starting post-processing")
        final_result_path = fox.get_result_path()
        command = f'{args.post_processing} "{final_result_path}"'
        print(f"calling '{command}'")
        system(command)


if __name__ == "__main__":
    main()
