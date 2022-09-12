from tokenization_utils import *
from tree_sitter import Language, Parser

import matplotlib.pyplot as plt
import networkx as nx
import pydot
# from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout
import copy
import random
import pickle

def convert_start_end(code, position):
    """
    use char level index as start and end
    """
    start, end = [position[0], position[1]], [position[2], position[3]]
    new_start, new_end = 0, 0
    lines = code.split('\n')
    start_line_ind = start[0]
    end_line_ind = end[0]
    pre_lines, post_lines = "", ""
    if start_line_ind > 0:
        pre_lines = "\n".join(lines[:start_line_ind]) + "\n"
    if end_line_ind > 0:
        post_lines = "\n".join(lines[:end_line_ind]) + "\n"
    new_start = len(pre_lines + lines[start_line_ind][:start[1]])
    new_end = len(post_lines + lines[end_line_ind][:end[1]])
    return new_start, new_end

# No matter what's the language, keep the original indent and new_line in snippet
def print_snippet(code, start, end, lang):
    snippet = code[start:end]
    indent = check_indent(code, start)
    new_line = check_new_line(code, end, lang)
    # instead of adding indent, how to add INDENT, DEDENT and NEW_LINE?
    if indent:
        snippet = indent + snippet
    if new_line:
        snippet += new_line
    elif check_space(code, end):
        snippet += " "
    return snippet
    
def print_snippet_root(code, root_node):
    start, end = root_node.start_point, root_node.end_point
    return print_start_end_util(code, start, end)

def print_snippet_pos(code, position):
    start, end = [position[0], position[1]], [position[2], position[3]]
    return print_start_end_util(code, start, end)

def print_start_end_util(code, start, end):
    lines = code.split('\n')
    start_line_ind = start[0]
    end_line_ind = end[0]
    snippet = ""
    if start_line_ind != end_line_ind:
        start_line = lines[start_line_ind][start[1]:]
        end_line = lines[end_line_ind][:end[1]]
        snippet = "\n".join([start_line] + lines[start_line_ind+1:end_line_ind] + [end_line])
    else:
        snippet = lines[start_line_ind][start[1]:end[1]]
    return snippet

def check_indent(code, start):
    snippet = code[:start]
    lines = snippet.split('\n')
    last_line = lines[-1]
    if last_line.isspace():
        return last_line
    return False

def check_space(code, end):
    char = code[end:][0]
    if char == " ":
        return True
    return False
    

# 应该加上token后面的space
def check_new_line(code, end, lang):
    snippet = code[end:].lstrip(' ')
    if snippet == "":
        return "\n"
    char = snippet[0]
    if char == '\n':
        return "\n"
    if lang == 'Python':
        if char == ';':
            new_snippet = snippet[1:]
            new_line = check_new_line(new_snippet, 0, lang)
            if new_line:
                return " ;" + new_line
            else:
                return " ; "
            return ";"
    return False

# Convert original tree-sitter graph to new graph
def get_graph(code, tree, lang):
    '''
    Convert original tree-sitter tree to new tree
    '''
    queue = []
    graph = {}
    depth = 0
    is_root = True
    root = ""
    try:
        queue.append((tree, depth, max_depth(tree, depth), None))
    except:
        return root, graph
    while len(queue) > 0:
        node, cur_depth, max_dep, parent = queue.pop(0)
        node_dict = {}
        node_dict["depth"] = cur_depth
        node_dict["max_depth"] = max_dep
        node_dict["height"] = max_dep-cur_depth
        node_dict["position"] = get_node_pos(node)
        new_start, new_end = convert_start_end(code, node_dict["position"])
        node_dict["start_end"] = (new_start, new_end)
#         snippet = print_start_end(code, node)
#         assert(code[new_start:new_end] == snippet)
        node_dict["snippet"] = print_snippet(code, new_start, new_end, lang)
        node_dict["id"] = node.type + "_" + str(get_node_id(node))
        if parent:
            node_dict["parent"] = parent.type + "_" + str(get_node_id(parent))
        else:
            node_dict["parent"] = ""
        node_dict["is_leaf"] = False
        if len(node.children) == 0:
            node_dict["is_leaf"] = True
#         -----------------
#         if node_dict["is_leaf"]:
#             print(node_dict["snippet"].replace(" ", "_"))
#         -----------------
        node_dict["type"] = node.type
        node_dict["label"] = node.type
        
        if len(node.children) == 0:
            node_dict["children"] = []
            node_dict["label"] = node_dict["snippet"]
            graph[node_dict["id"]] = node_dict
            continue
        node_dict["children"] = []
        for child in node.children:
            depth = cur_depth + 1
            node_dict["children"].append(child.type + "_" + str(get_node_id(child)))
            try:
                queue.append((child, depth, max_depth(child, depth), node))
            except:
                return "", {}
        graph[node_dict["id"]] = node_dict
        if is_root:
            root = node_dict["id"]
            is_root = False
    return root, graph

def pipeline(code, parser, lang, dep_cap_k=3):
    root_node = get_ast(code, parser)
    root, graph = get_graph(code, root_node, lang)
    root, graph_pruned = get_graph_pruned(root, graph, dep_cap=dep_cap_k)
    root, graph_sibs = get_graph_merge_sibs(root, graph)
    root, graph_pruned_sibs = get_graph_merge_sibs(root, graph_pruned)
    return root, graph, graph_pruned, graph_sibs, graph_pruned_sibs

def refine_graphs(root, graph, dep_cap_k=3):
#     root_node = get_ast(code, parser)
#     root, graph = get_graph(code, root_node)
    root, graph_pruned = get_graph_pruned(root, graph, dep_cap=dep_cap_k)
    root, graph_sibs = get_graph_merge_sibs(root, graph)
    root, graph_pruned_sibs = get_graph_merge_sibs(root, graph_pruned)
    return root, graph, graph_pruned, graph_sibs, graph_pruned_sibs


def tree_to_code_bfs(root, graph, code):
    nodes = []
    queue = []
    queue.append(root)
    while len(queue) > 0:
        node_id = queue.pop(0)
        node = graph[node_id]
        if node['is_leaf']:
            nodes.append(node['label'])
        for child in node["children"]:
            queue.append(child)
    print(code)
    print(" ".join(nodes))
    return

def traverse_tree_dfs(root, graph, nodes):
    node = graph[root]
    if len(node["children"]) == 0:
#         print(node['label'], len(node['label']))
        nodes.append(node['label'])
    for child in node["children"]:
        traverse_tree_dfs(child, graph, nodes)
    return

def format_tree_to_code(nodes):
    for node in nodes:
        node = node.replace("    ", "INDENT")
    code = " ".join(nodes)
    lines = code.split('\n')
    code = "\n".join([x.strip() for x in lines])
    code = code.replace("INDENT", "    ")
    return code

# for python, the space in " ".join can cause problem.
def tree_to_code_dfs(root, graph):
    """
    convert ast nodes to code recursively
    """
    nodes = []
    traverse_tree_dfs(root, graph, nodes)
    code = "".join(nodes)
#     code = format_tree_to_code(nodes)
    return code

def tree_to_code(root, graph, lang):
    """
    convert ast graph back to code
    """
    parsed_code = tree_to_code_dfs(root, graph)
    return parsed_code
#     return detok_format(parsed_code, file_detokenizers[lang])

def traverse_tree_bfs(root, graph):
    '''
    traverse a tree-sitter tree in bfs manner. Sort by the depth of each node.
    '''
    queue = []
    dep_dict = {}
    queue.append(root)
    while len(queue) > 0:
        node_id = queue.pop(0)
        node = graph[node_id]
        if node['depth'] in dep_dict:
            dep_dict[node['depth']].append(node)
        else:
            dep_dict[node['depth']] = [node]
        for child in node["children"]:
            queue.append(child)
    return dep_dict
#     return detok_format(code, file_detokenizers[lang])
    

def get_ast_seq_bfs(root, graph):
    """
    flatten ast to a sequence
    """
    sep_1 = "<SEP>"
    nt_bs = "<SPACE>"
    nt_prefix = "$"
    # get the nodes by depth. The nodes at the same depth are in original order
    dep_dict = traverse_tree_bfs(root, graph)
    dep_list = sorted(list(dep_dict.keys()))
    seq_list = []
    for dep in dep_list:
        node_list = dep_dict[dep]
        labels = []
        parent = node_list[0]['parent']
        parent_id = 0
        for node in node_list:
            new_parent = node['parent']
            # when the subtree changes, we need to note it down. Do we really need to?
            if new_parent != parent:
                parent = new_parent
                parent_id += 1
            label = node['label']
            if not node['is_leaf']:
                label = nt_prefix + node['label']
            labels.append(label + "_" + str(parent_id) + nt_bs)
        seq = "".join(labels)
        seq_list.append(seq)
    code_seq = sep_1.join(seq_list)
    return code_seq

def get_ast_seq_bfs_subsep(root, graph):
    """
    flatten ast to a sequence
    """
    sep_1 = "<SEP>"
    sep_2 = "<SUBSEP>"
    nt_bs = "<SPACE>"
    nt_prefix = "$"
    # get the nodes by depth. The nodes at the same depth are in original order
    dep_dict = traverse_tree_bfs(root, graph)
    dep_list = sorted(list(dep_dict.keys()))
    seq_list = []
    for dep in dep_list:
        node_list = dep_dict[dep]
        labels = []
        parent = node_list[0]['parent']
        for node in node_list:
            new_parent = node['parent']
            # when the subtree changes, we need to note it down. Do we really need to?
            if new_parent != parent:
                parent = new_parent
                labels.append(sep_2)
            label = node['label']
            if not node['is_leaf']:
                label = nt_prefix + node['label']
            labels.append(label + nt_bs)
        seq = "".join(labels)
        seq_list.append(seq)
    code_seq = sep_1.join(seq_list)
    return code_seq

def code_seq_remove_subseq(code_seq):
    sep_1 = "<SEP>"
    sep_2 = "<SUBSEP>"
    nt_bs = "<SPACE>"
    nt_prefix = "$"
    seq_list = code_seq.split(sep_1)
    new_seq_list = []
    for dep, seq in enumerate(seq_list):
        subtree_list = seq.split(sep_2)
        new_subtree_list = []
        for i, subtree in enumerate(subtree_list):
            if len(subtree) == 0:
                print("subtree", subtree_list)
                new_subtree_list.append("")
                continue
            node_list = subtree.split(nt_bs)
            new_node_list = []
            parent = i
            for node in node_list:
                if len(node) == 0:
                    continue
                nodestr = node + "_" + str(parent)
                new_node_list.append(nodestr)
            new_subtree = nt_bs.join(new_node_list)
            new_subtree_list.append(new_subtree)
        new_seq = nt_bs.join(new_subtree_list)
        new_seq_list.append(new_seq)
    new_code_seq = sep_1.join(new_seq_list)
    return new_code_seq


def code_seq_add_subseq(code_seq):
    return

import re

def merge_node_list(node_list):
    """
    for a list of nodes, merge the ones that's not ending with "_[0-9]+"
    So that we don't split in the middle of a node
    """
    new_node_list = []
    cur_node = []
    for node in node_list:
        cur_node += [node]
        x = re.search('_[0-9]+$', node)
        if x:
            new_node_list.append(" ".join(cur_node))
            cur_node = []
    return new_node_list


# Why not convert dep_dict with child-parent edges to the original code first?
# Is dep_dict alone enough? Probably not.
# Do I have to convert dep_dict with child-parent edges to graph and then to code?
# I think so.

def get_node_id_alt(node, depth, index):
    if node == "":
        return node
    return node + "_" + str(depth) + "_" + str(index)

def dep_dict_to_ast(dep_dict, parent_dict):
    """
    construct ast from dep_dict with child-parent edges
    """
    dep_list = sorted(list(dep_dict.keys()))
    graph = {}
    root = ""
    nt_prefix = "$"
    for dep in dep_list:
        node_list = dep_dict[dep]
        nt_idx = 0
        term_idx = 0
        for node, parent in node_list:
            node_dict = {}
            node_dict['depth'] = dep
            node_dict['label'] = node
            node_dict['children'] = []
            if node.startswith(nt_prefix):
                node_dict['is_leaf'] = False
                node_dict['id'] = get_node_id_alt(node, dep, nt_idx)
                nt_idx += 1
            else:
                node_dict['is_leaf'] = True
                node_dict['id'] = get_node_id_alt(node, dep, term_idx)
                term_idx += 1
            if dep == 0:
                root = node_dict['id']
                node_dict['parent'] = ""
            else:
                node_parent, _ = parent_dict[dep-1][int(parent)]
                node_dict['parent'] = get_node_id_alt(node_parent, dep-1, int(parent))
                graph[node_dict['parent']]['children'].append(node_dict['id'])
            graph[node_dict['id']] = node_dict
    return root, graph
    

def ast_seq_to_ast(seq):
    """
    first of all, the flattened_ast is from dep_dict. So can we construct the graph from dep_dict?
    In dep_dict, what we have lost? The edges.
    How do we keep the edges? By adding parents to each node.
    """
    sep_1 = "<SEP>"
    nt_bs = "<SPACE>"
    nt_prefix = "$"
    seq_list = seq.split(sep_1)
    dep_dict = {}
    graph = {}
    parent_dict = {}
    for dep, seq in enumerate(seq_list):
        dep_dict[dep] = []
        parent_dict[dep] = []
        node_list = seq.split(nt_bs)
#         node_list = merge_node_list(node_list)
        for nodestr in node_list:
            if len(nodestr) == 0:
                continue
            node, parent = nodestr.rsplit('_', 1)
            if nodestr.startswith(nt_prefix):
                parent_dict[dep].append((node, parent))
            dep_dict[dep].append((node, parent))
    root, graph = dep_dict_to_ast(dep_dict, parent_dict)
    return root, graph
    
def ast_seq_to_code(seq, lang):
    root, graph = ast_seq_to_ast(seq)
    code = tree_to_code(root, graph, lang)
    return code
# 问题：
# 只有一个孩子的节点应该被flatten吗？




def show_graph(root, graph):
    edge_list, labels = get_graph_data_new(root, graph)
    show_graph_from_edge(edge_list, labels)
    return

def show_graph_from_edge(edge_list, labels):
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    nx.nx_agraph.write_dot(G,'test.dot')
    pos=graphviz_layout(G, prog='dot')
    plt.figure(1,figsize=(25,25)) 
    nx.draw(G, pos, with_labels=False, arrows=False)
    nx.draw_networkx_labels(G,pos,labels,font_size=10,font_color='black')
    plt.show()
    return





def get_terms(root, graph, term_dict):
    queue = []
    queue.append(root)
    while len(queue) > 0:
        node_id = queue.pop(0)
        node = graph[node_id]
        if not node['is_leaf']:
            if node['type'] in term_dict:
                term_dict[node['type']] += 1
            else:
                term_dict[node['type']] = 1
        for child in node["children"]:
            queue.append(child)
    return term_dict

def get_all_terms(jsons, tag):
    if os.path.exists("term_dict.pkl"):
        with open("term_dict.pkl", 'rb') as infile:
            term_dict_lang = pickle.load(infile)
        with open("sub_term_dict.pkl", 'rb') as infile:
            sub_term_dict = pickle.load(infile)
        return term_dict_lang, sub_term_dict
    term_dict_lang = {}
#     tag = 'train'
    for lang in langs:
        term_dict = {}
        key = ('desc', lang)
        for k, code in tqdm(jsons[key][tag].items()):
            root_node = get_ast(code, ast_parsers[lang])
            root, graph = get_graph(code, root_node, lang)
            term_dict = get_terms(root, graph, term_dict)
        term_dict_lang[lang] = term_dict
    
    sub_term_dict = {}
    for lang in langs:
        sub_t_dict = {}
        t_dict = term_dict_lang[lang]
        for k, v in t_dict.items():
            last_token = k
            if "_" in k:
                sub_tokens = k.split('_')
                last_token = sub_tokens[-1]
            if last_token in sub_t_dict:
                sub_t_dict[last_token] += v
            else:
                sub_t_dict[last_token] = v
        sub_term_dict[lang] = sub_t_dict
    
    with open("term_dict.pkl", 'wb') as outfile:
        pickle.dump(term_dict_lang, outfile)
    with open("sub_term_dict.pkl", 'wb') as outfile:
        pickle.dump(sub_term_dict, outfile)
    return term_dict_lang, sub_term_dict


def traverse_tree_recur(tree):
    '''
    recursively traverse tree-sitter tree
    '''
    if tree:
        if len(tree.children) == 0:
            print(tree.type)
        for child in tree.children:
            traverse_tree_recur(child)
    return

def traverse_tree_new(root, graph):
    '''
    recursively traverse new tree
    '''
    node = graph[root]
    if len(node["children"]) == 0:
        print(node['label'])
    for child in node["children"]:
        traverse_tree_new(child, graph)
    return

def max_depth(tree, depth):
    '''
    get the max depth of a given tree/subtree
    '''
    max_dep_list = [depth]
    for child in tree.children:
        max_dep_list.append(max_depth(child, depth + 1))
    return max(max_dep_list)

def max_depth_new(root, graph, depth):
    '''
    get the max depth of a given tree/subtree
    '''
    node_dict = graph[root]
    max_dep_list = [depth]
    for child in node_dict["children"]:
        max_dep_list.append(max_depth_new(child, graph, depth + 1))
    return max(max_dep_list)



def update_depth(root, graph):
    queue = []
    depth = 0
    queue.append((root, depth, max_depth_new(root, graph, depth)))
    while len(queue) > 0:
        node_id, cur_depth, max_dep = queue.pop(0)
        node_dict = graph[node_id]
        node_dict["depth"] = cur_depth
        node_dict["max_depth"] = max_dep
        if node_dict["height"] == 0:
            assert max_dep - cur_depth == 0
        node_dict["height"] = max_dep - cur_depth
        for child in node_dict["children"]:
            depth = cur_depth + 1
            queue.append((child, depth, max_depth_new(child, graph, depth)))
    return

def get_graph_pruned(root, graph, dep_cap=0):
    queue = []
    depth = 0
    queue.append((root, depth, max_depth_new(root, graph, depth)))
    new_graph = {}
    while len(queue) > 0:
        node_id, cur_depth, max_dep = queue.pop(0)
        node_dict = copy.deepcopy(graph[node_id])
        if max_dep-cur_depth < dep_cap:
            node_dict['label'] = node_dict['snippet']
            node_dict["height"] = 0
            node_dict['is_leaf'] = True
            node_dict['children'] = []
            new_graph[node_id] = node_dict
            continue
        new_graph[node_id] = node_dict
        for child in node_dict["children"]:
            depth = cur_depth + 1
            queue.append((child, depth, max_depth_new(child, graph, depth)))
    update_depth(root, new_graph)
    return root, new_graph

def get_graph_data_new(root, graph):
    '''
    get the edges, labels from the new graph
    '''
    queue = []
    edge_list = []
    labels = {}
    queue.append(root)
    new_graph = {}
    while len(queue) > 0:
        node_id = queue.pop(0)
        node_dict = graph[node_id]
        parent_id = node_dict["parent"]
        if parent_id != "":
            edge = (parent_id, node_id)
            edge_list.append(edge)
                        
        for i, child in enumerate(node_dict["children"]):
            queue.append(child)
        labels[node_id] = node_dict['label']
    return edge_list, labels
    
    
def need_merge_sib(children, graph):
    merged_children = []
    chunks = []
    chunk = []
    
    for i, child in enumerate(children):
        if graph[child]['is_leaf']:
            chunk.append(i)
        else:
            if len(chunk) > 0:
                chunks.append(chunk)
            chunks.append([i])
            chunk = []
    if len(chunk) > 0:
        chunks.append(chunk)
    for chunk in chunks:
        if len(chunk) > 1:
            return chunks
    return None
    
def chunk_children(chunks, children, graph):
    merged_children = []
    new_children = []
    
    for chunk in chunks:
        if len(chunk) == 1:
            merged_children.append(children[chunk[0]])
            continue
        siblings = [children[x] for x in chunk]
        merge_node = merge_sibs(siblings, graph)
        merged_children.append(merge_node['id'])
        new_children.append(merge_node)
    return new_children, merged_children
    
def merge_sibs(children, graph):
    snippets = []
    positions = []
    merge_type = "merged_sibs"
    for child in children:
        snippets.append(graph[child]['snippet'])
        positions.append(graph[child]['position'])
    merge_node = copy.deepcopy(graph[children[0]])
    merge_node["position"] = positions[0][:2] + positions[-1][2:]
    merge_node["type"] = merge_type
    merge_node["id"] = merge_type + "_" + "_".join([str(x) for x in merge_node["position"]])
    merge_node["snippet"] = "".join(snippets)
    merge_node["label"] = merge_node["snippet"]
    return merge_node


def get_graph_merge_sibs(root, graph):
    queue = []
    depth = 0
    queue.append(root)
    new_graph = {}
    while len(queue) > 0:
        node_id = queue.pop(0)
        if node_id in new_graph:
            continue
        node_dict = copy.deepcopy(graph[node_id])
        children = node_dict["children"]
        chunks = need_merge_sib(children, graph)
        if chunks:
            new_children, merged_children = chunk_children(chunks, children, graph)
            node_dict['children'] = merged_children
            for new_child in new_children:
                new_graph[new_child['id']] = new_child
        new_graph[node_id] = node_dict
        for child in node_dict["children"]:
            queue.append(child)
    return root, new_graph

# depth-first traverse
def traverse_tree(tree):
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        for child in cursor.node.children:
            print("cr", child.type, cursor.node.type)
        yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False
    return

# tree = traverse_tree(java_root_node)
# for node in tree:
#     print(node.type)

def traverse_tree_recur(tree):
    if tree:
        if len(tree.children) == 0:
            print(tree.type)
        for child in tree.children:
            traverse_tree_recur(child)
    return

def get_node_id(tree):
    start, end = tree.start_point, tree.end_point
    node_id = str(start[0]) + "_" + str(start[1])+ "_" + str(end[0])+ "_" + str(end[1])
    return node_id

def get_node_pos(tree):
    start, end = tree.start_point, tree.end_point
    return [start[0], start[1], end[0], end[1]]


def get_ast(code, parser):
    root_node = parser.parse(bytes(code, "utf8")).root_node
    return root_node

def check_next_token(next_token, depth, space, brack_stack):
    ind_start = next_token.find(')')
    num_right_brac = len(next_token[ind_start:])
    
    if ind_start < len(next_token)-1:
        if set(next_token[ind_start:]) == set(")"):
            new_next_tok = next_token[:ind_start+1]
            if not next_token.startswith("("):
                new_next_tok = next_token[:ind_start]
                num_right_brac += 1
            print("".join((depth)*[space]), new_next_tok)
            assert(len(brack_stack) >= num_right_brac-1)
            for i in range(num_right_brac-1):
                depth = brack_stack.pop()
                print("".join((depth)*[space]), ")")
        else:
            print("exp1!")
    else:
        print("".join((depth)*[space]), next_token)
    return depth

def parse_sexp_str(sexp_str):
    tokens = sexp_str.split()
    print(tokens)
    depth = 0
    i = 0
    space = "  "
    brack_stack = []
    while i < len(tokens):
        token = tokens[i]
        next_token = None
        if i < len(tokens)-1:
            next_token = tokens[i+1]
        if token[0] == "(" and token[-1] != ":" and token[-1] != ")":
            print("".join(depth*[space]),"(")
            brack_stack.append(depth)
            depth += 1
            print("".join(depth*[space]), token[1:])
            depth += 1
            i += 1

        elif token[-1] == ":":
            print("".join((depth)*[space]), token)
            i += 1         

        elif token[-1]==")": #token[0]=="(" and 
            depth = check_next_token(token, depth, space, brack_stack)
            i += 1
        else:
            print("exp!", token)
            break
    return

def count_char(code_str):
    char_dict = {}
    for c in code_str:
        if c in char_dict:
            char_dict[c] += 1
        else:
            char_dict[c] = 1
    return char_dict

def get_all_sub_trees(root_node):
    node_stack = []
    sub_tree_sexp_list = []
    depth = 1
    node_stack.append([root_node, depth])
    while len(node_stack) != 0:
        cur_node, cur_depth = node_stack.pop()
        sub_tree_sexp_list.append([cur_node.sexp(), cur_depth])
        for child_node in cur_node.children:
            if len(child_node.children) != 0:
                depth = cur_depth + 1
                node_stack.append([child_node, depth])
    return sub_tree_sexp_list

# home_path = "/home/mingzhu/CodeModel/"
# data_path = home_path + "g4g/XLCoST_data/"
# so_path = "/home/mingzhu/CodeModel/CodeGen/codegen_sources/preprocessing/lang_processors/"

CPP_LANGUAGE = Language(so_path + 'my-languages.so', 'cpp')
cpp_parser = Parser()
cpp_parser.set_language(CPP_LANGUAGE)

JAVA_LANGUAGE = Language(so_path + 'my-languages.so', 'java')
java_parser = Parser()
java_parser.set_language(JAVA_LANGUAGE)

PY_LANGUAGE = Language(so_path + 'my-languages.so', 'python')
py_parser = Parser()
py_parser.set_language(PY_LANGUAGE)

CS_LANGUAGE = Language(so_path + 'my-languages.so', 'c_sharp')
cs_parser = Parser()
cs_parser.set_language(CS_LANGUAGE)

JS_LANGUAGE = Language(so_path + 'my-languages.so', 'javascript')
js_parser = Parser()
js_parser.set_language(JS_LANGUAGE)

PHP_LANGUAGE = Language(so_path + 'my-languages.so', 'php')
php_parser = Parser()
php_parser.set_language(PHP_LANGUAGE)

C_LANGUAGE = Language(so_path + 'my-languages.so', 'c')
c_parser = Parser()
c_parser.set_language(C_LANGUAGE)

ast_parsers = {"Java": java_parser, "C++": cpp_parser, "C": c_parser, "Python": py_parser, "Javascript": js_parser,
                   "PHP": php_parser, "C#":cs_parser}



