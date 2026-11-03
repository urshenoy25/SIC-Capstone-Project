"""
Module 2B: Causal Directed Acyclic Graph (DAG) Engine
Constructs, validates, and visualizes the Structural Causal DAG for Telco Churn.
Enforces domain tier constraints, acyclicity, and topological ordering for downstream SCM propagation.
"""

import json
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional


# Domain Tier Definitions (Strict hierarchical ordering)
# Tier 0: Exogenous Demographics (no incoming edges from operational features)
# Tier 1: Core Service Foundations
# Tier 2: Dependent Digital Add-on Services
# Tier 3: Commercial & Billing Terms
# Tier 4: Financial Aggregates
# Tier 5: High-Stakes Decision / Outcome (No outgoing edges allowed)

FEATURE_TIERS: Dict[str, int] = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 0,

    "PhoneService": 1,
    "InternetService": 1,

    "MultipleLines": 2,
    "OnlineSecurity": 2,
    "OnlineBackup": 2,
    "DeviceProtection": 2,
    "TechSupport": 2,
    "StreamingTV": 2,
    "StreamingMovies": 2,

    "Contract": 3,
    "PaperlessBilling": 3,
    "PaymentMethod": 3,

    "MonthlyCharges": 4,
    "TotalCharges": 4,

    "Churn": 5
}


class TelcoCausalGraph:
    """
    Manages the Directed Acyclic Graph (DAG) for customer churn dynamics.
    Enforces causal hierarchy, validates absence of cycles, and provides
    parent-child queries for structural equation fitting and do-calculus.
    """

    def __init__(self):
        self.tiers = FEATURE_TIERS
        self.dag = nx.DiGraph()
        self._build_domain_dag()

    def _build_domain_dag(self):
        """Constructs the domain causal DAG reflecting real-world business and technical mechanics."""
        # 1. Add all nodes with metadata
        for node, tier in self.tiers.items():
            self.dag.add_node(
                node,
                tier=tier,
                node_type="target" if node == "Churn" else ("immutable" if tier == 0 else "mutable")
            )

        # 2. Add structural causal edges (Cause -> Effect)
        causal_edges: List[Tuple[str, str]] = [
            # Demographics influence services & baseline usage
            ("SeniorCitizen", "InternetService"),
            ("SeniorCitizen", "TechSupport"),
            ("Partner", "Dependents"),
            ("Partner", "MultipleLines"),
            ("Dependents", "StreamingTV"),
            ("Dependents", "StreamingMovies"),

            # Core Services trigger add-ons and multiple lines
            ("PhoneService", "MultipleLines"),
            ("InternetService", "OnlineSecurity"),
            ("InternetService", "OnlineBackup"),
            ("InternetService", "DeviceProtection"),
            ("InternetService", "TechSupport"),
            ("InternetService", "StreamingTV"),
            ("InternetService", "StreamingMovies"),
            ("InternetService", "MonthlyCharges"),

            # Service Add-ons directly impact monthly billing and loyalty
            ("OnlineSecurity", "MonthlyCharges"),
            ("OnlineSecurity", "Churn"),
            ("OnlineBackup", "MonthlyCharges"),
            ("DeviceProtection", "MonthlyCharges"),
            ("TechSupport", "MonthlyCharges"),
            ("TechSupport", "Churn"),
            ("StreamingTV", "MonthlyCharges"),
            ("StreamingMovies", "MonthlyCharges"),
            ("MultipleLines", "MonthlyCharges"),

            # Commercial terms govern pricing, churn vulnerability, and payment channel
            ("Contract", "MonthlyCharges"),
            ("Contract", "Churn"),
            ("PaperlessBilling", "PaymentMethod"),
            ("PaymentMethod", "Churn"),

            # Tenure & Billing govern accumulated cost and churn risk
            ("tenure", "Contract"),
            ("tenure", "TotalCharges"),
            ("tenure", "Churn"),
            ("MonthlyCharges", "TotalCharges"),
            ("MonthlyCharges", "Churn"),
            ("TotalCharges", "Churn")
        ]

        for u, v in causal_edges:
            self.add_edge(u, v)

        assert nx.is_directed_acyclic_graph(self.dag), "Error: Causal graph contains cyclical dependencies!"

    def add_edge(self, source: str, target: str):
        """Adds directed edge while strictly enforcing tier hierarchy (no reverse causality)."""
        source_tier = self.tiers.get(source, 99)
        target_tier = self.tiers.get(target, 99)

        if source_tier > target_tier:
            raise ValueError(
                f"Forbidden reverse causality: Node '{source}' (Tier {source_tier}) cannot cause '{target}' (Tier {target_tier})"
            )
        if target == "Churn" and source_tier >= 5:
            raise ValueError("Target 'Churn' cannot have intra-target or outgoing edges.")

        self.dag.add_edge(source, target)

    def is_acyclic(self) -> bool:
        """Returns True if graph is a valid Directed Acyclic Graph."""
        return nx.is_directed_acyclic_graph(self.dag)

    def get_parents(self, node: str) -> List[str]:
        """Returns direct causal parents PA_j of a node."""
        return sorted(list(self.dag.predecessors(node)))

    def get_children(self, node: str) -> List[str]:
        """Returns direct causal children of a node."""
        return sorted(list(self.dag.successors(node)))

    def get_ancestors(self, node: str) -> Set[str]:
        """Returns all ancestral nodes that have a causal path leading to node."""
        return nx.ancestors(self.dag, node)

    def get_descendants(self, node: str) -> Set[str]:
        """Returns all descendant nodes that are affected by an intervention on node."""
        return nx.descendants(self.dag, node)

    def topological_sort(self) -> List[str]:
        """Returns nodes in topological causal execution order."""
        return list(nx.topological_sort(self.dag))

    def export_graph_dict(self) -> Dict[str, Any]:
        """Exports DAG nodes and edges in serialized format for frontend UI and REST API."""
        nodes = []
        for n, data in self.dag.nodes(data=True):
            nodes.append({
                "id": n,
                "label": n,
                "tier": data.get("tier", 0),
                "type": data.get("node_type", "mutable")
            })

        edges = []
        for u, v in self.dag.edges():
            edges.append({
                "from": u,
                "to": v,
                "weight": 1.0
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "is_acyclic": self.is_acyclic(),
            "topological_order": self.topological_sort()
        }

    def to_dot(self) -> str:
        """Exports Graphviz DOT string for static visual rendering."""
        dot_lines = ["digraph TelcoCausalDAG {", "  rankdir=LR;", "  node [fontname=Arial, fontsize=11];"]
        for u, v in self.dag.edges():
            dot_lines.append(f'  "{u}" -> "{v}";')
        dot_lines.append("}")
        return "\n".join(dot_lines)


if __name__ == "__main__":
    cg = TelcoCausalGraph()
    print("=" * 65)
    print("MODULE 2B: CAUSAL DIRECTED ACYCLIC GRAPH (DAG)")
    print("=" * 65)
    print(f"Total Graph Nodes : {cg.dag.number_of_nodes()}")
    print(f"Total Graph Edges : {cg.dag.number_of_edges()}")
    print(f"Is strictly DAG   : {cg.is_acyclic()}")
    print(f"\nTopological Sort Order:\n{cg.topological_sort()}")
    print("\nCausal Parents of 'Churn':", cg.get_parents("Churn"))
    print("Causal Children of 'Contract':", cg.get_children("Contract"))
    print("Causal Descendants of 'InternetService':", cg.get_descendants("InternetService"))
    print("=" * 65)
