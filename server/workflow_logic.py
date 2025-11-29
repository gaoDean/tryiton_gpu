import random
from utils import logger

def get_dynamic_mappings(workflow):
    """
    Scans the workflow JSON to automatically find node IDs.
    Returns a dictionary of critical Node IDs.
    """
    mappings = {
        "target_image_node": None,
        "ref_image_node": None,
        "sampler_nodes": [],
        "save_image_node": None
    }
    
    load_image_nodes = []

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        # 1. Find Load Image Nodes
        if class_type == "LoadImage":
            load_image_nodes.append(node_id)

        # 2. Find Sampler Nodes (Look for Hair classes or KSampler with a seed)
        if class_type in ["ApplyHairTransfer", "ApplyHairRemover", "KSampler", "KSamplerAdvanced"]:
            if "seed" in inputs:
                mappings["sampler_nodes"].append(node_id)

        # 3. Find Save Image Node
        if class_type == "SaveImage":
            mappings["save_image_node"] = node_id

    # Sort LoadImage nodes by ID to ensure strict order
    # Node 1 -> Target, Node 2/10 -> Reference
    load_image_nodes.sort(key=lambda x: int(x))

    if len(load_image_nodes) >= 2:
        mappings["target_image_node"] = load_image_nodes[0]
        mappings["ref_image_node"] = load_image_nodes[1]
    else:
        raise ValueError(f"Found {len(load_image_nodes)} LoadImage nodes. Needed 2.")

    if not mappings["save_image_node"]:
        raise ValueError("No SaveImage node found.")

    return mappings

def prepare_workflow(workflow, mappings, target_filename, ref_filename, options=None):
    """
    Injects image paths and random seeds into the workflow dictionary.
    """
    if options is None:
        options = {}

    # Helper to format paths for ComfyUI
    def get_node_path(filename):
        return filename # Assumes filename includes subfolder if applicable

    # Inject Images
    workflow[mappings["target_image_node"]]["inputs"]["image"] = target_filename
    workflow[mappings["ref_image_node"]]["inputs"]["image"] = ref_filename
    
    # Inject Random Seeds
    for node_id in mappings["sampler_nodes"]:
        if node_id in workflow and "inputs" in workflow[node_id]:
            seed = random.randint(1, 10**14)
            workflow[node_id]["inputs"]["seed"] = seed
            logger.info(f"🎲 Seed for Node {node_id} set to: {seed}")
    
    # Inject Optional Parameters
    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        
        if class_type == "ApplyHairRemover":
            if options.get("remover_steps") is not None:
                node["inputs"]["steps"] = options["remover_steps"]
            if options.get("remover_strength") is not None:
                node["inputs"]["strength"] = options["remover_strength"]
            if options.get("remover_cfg") is not None:
                node["inputs"]["cfg"] = options["remover_cfg"]
        
        elif class_type == "ApplyHairTransfer":
            if options.get("transfer_steps") is not None:
                node["inputs"]["steps"] = options["transfer_steps"]
            if options.get("transfer_cfg") is not None:
                node["inputs"]["cfg"] = options["transfer_cfg"]
            if options.get("transfer_control_strength") is not None:
                node["inputs"]["control_strength"] = options["transfer_control_strength"]
            if options.get("transfer_adapter_strength") is not None:
                node["inputs"]["adapter_strength"] = options["transfer_adapter_strength"]

    return workflow