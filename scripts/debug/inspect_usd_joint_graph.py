#!/usr/bin/env python3
"""Inspect parent-child relationships of joints in a USD robot asset.

Examples:

    conda run -n env_isaaclab python scripts/debug/inspect_usd_joint_graph.py
    conda run -n env_isaaclab python scripts/debug/inspect_usd_joint_graph.py --all
    conda run -n env_isaaclab python scripts/debug/inspect_usd_joint_graph.py --pattern "hand_camera_base|hand_palm|wrist_yaw"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD_PATH = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "g1-29dof_wholebody_inspire"
    / "g1_29dof_with_inspire_rev_1_0_no_hand_camera.usd"
)
DEFAULT_PATTERN = "hand_camera_base|hand_palm|hand_base|wrist_yaw"


@dataclass(frozen=True)
class JointEdge:
    """A directed body0 -> body1 edge parsed from a USD PhysicsJoint."""

    joint_path: str
    joint_name: str
    joint_type: str
    body0: str
    body1: str
    body0_active: bool
    body1_active: bool


def _load_usd_module():
    """Import pxr.Usd with a clear error when the Isaac Sim Python env is missing."""

    try:
        from pxr import Usd
    except ImportError as exc:
        raise SystemExit(
            "Failed to import pxr.Usd. Run this with the Isaac Sim/Isaac Lab Python environment, for example:\n"
            "  conda run -n env_isaaclab python scripts/debug/inspect_usd_joint_graph.py"
        ) from exc
    return Usd


def _resolve_path(path: Path) -> Path:
    """Resolve relative paths from cwd first, then from the repository root."""

    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def _relationship_target(prim, relationship_name: str) -> str | None:
    """Return the first target path for a USD relationship."""

    relationship = prim.GetRelationship(relationship_name)
    if not relationship:
        return None
    targets = relationship.GetTargets()
    if not targets:
        return None
    return str(targets[0])


def _joint_edges(stage) -> list[JointEdge]:
    """Collect USD PhysicsJoint edges from the stage."""

    edges: list[JointEdge] = []
    for prim in stage.Traverse():
        joint_type = prim.GetTypeName()
        if "Joint" not in joint_type:
            continue
        body0 = _relationship_target(prim, "physics:body0")
        body1 = _relationship_target(prim, "physics:body1")
        if body0 is None or body1 is None:
            continue
        body0_prim = stage.GetPrimAtPath(body0)
        body1_prim = stage.GetPrimAtPath(body1)
        edges.append(
            JointEdge(
                joint_path=str(prim.GetPath()),
                joint_name=prim.GetName(),
                joint_type=joint_type,
                body0=body0,
                body1=body1,
                body0_active=body0_prim.IsValid() and body0_prim.IsActive(),
                body1_active=body1_prim.IsValid() and body1_prim.IsActive(),
            )
        )
    return edges


def _matches(edge: JointEdge, pattern: re.Pattern[str]) -> bool:
    """Return whether a joint edge matches a filter pattern."""

    text = "\n".join((edge.joint_path, edge.joint_name, edge.joint_type, edge.body0, edge.body1))
    return pattern.search(text) is not None


def _short(path: str) -> str:
    """Use the final prim name when printing relationship summaries."""

    return path.rsplit("/", 1)[-1]


def _print_edge(edge: JointEdge) -> None:
    """Print one joint relation in a compact parent-child form."""

    print(f"{edge.joint_name} [{edge.joint_type}]")
    print(f"  body0(parent): {edge.body0} active={edge.body0_active}")
    print(f"  body1(child) : {edge.body1} active={edge.body1_active}")
    print(f"  summary      : {_short(edge.body0)} --{edge.joint_name}--> {_short(edge.body1)}")


def _print_filtered_edges(edges: list[JointEdge], pattern: str, show_all: bool) -> None:
    """Print all selected joint edges."""

    selected_edges = edges if show_all else [edge for edge in edges if _matches(edge, re.compile(pattern))]
    print(f"[INFO] Found {len(edges)} joint edge(s) in USD.")
    print(f"[INFO] Printing {len(selected_edges)} joint edge(s).")
    if not show_all:
        print(f"[INFO] Filter pattern: {pattern}")
    print()
    for index, edge in enumerate(selected_edges):
        if index:
            print()
        _print_edge(edge)


def _incoming(edges: list[JointEdge], body_path: str) -> list[JointEdge]:
    """Return joints whose child body is body_path."""

    return [edge for edge in edges if edge.body1 == body_path]


def _outgoing(edges: list[JointEdge], body_path: str) -> list[JointEdge]:
    """Return joints whose parent body is body_path."""

    return [edge for edge in edges if edge.body0 == body_path]


def _print_body_branch(edges: list[JointEdge], body_name: str) -> None:
    """Print incoming and outgoing joints for bodies matching body_name."""

    body_paths = sorted({edge.body0 for edge in edges if edge.body0.endswith(body_name)})
    body_paths.extend(edge.body1 for edge in edges if edge.body1.endswith(body_name) and edge.body1 not in body_paths)
    if not body_paths:
        print(f"[WARN] No body path ending with {body_name!r} was found.")
        return

    for body_path in body_paths:
        print(f"\n[BRANCH] {body_path}")
        incoming_edges = _incoming(edges, body_path)
        outgoing_edges = _outgoing(edges, body_path)
        if incoming_edges:
            print("  incoming:")
            for edge in incoming_edges:
                print(
                    f"    {_short(edge.body0)} active={edge.body0_active} "
                    f"--{edge.joint_name} [{edge.joint_type}]--> "
                    f"{_short(edge.body1)} active={edge.body1_active}"
                )
        else:
            print("  incoming: none")
        if outgoing_edges:
            print("  outgoing:")
            for edge in outgoing_edges:
                print(
                    f"    {_short(edge.body0)} active={edge.body0_active} "
                    f"--{edge.joint_name} [{edge.joint_type}]--> "
                    f"{_short(edge.body1)} active={edge.body1_active}"
                )
        else:
            print("  outgoing: none")


def _print_hand_camera_assessment(edges: list[JointEdge]) -> None:
    """Summarize whether hand camera base links are separate fixed accessories."""

    print("\n[HAND CAMERA CHECK]")
    for side in ("left", "right"):
        wrist = next((edge.body1 for edge in edges if edge.joint_name == f"{side}_wrist_yaw_joint"), None)
        camera_edge = next((edge for edge in edges if edge.joint_name == f"{side}_hand_camera_base_joint"), None)
        hand_edge = next((edge for edge in edges if edge.joint_name == f"{side}_hand_palm_joint"), None)

        print(f"{side}:")
        if wrist is None:
            print("  wrist_yaw child link: missing")
        else:
            print(f"  wrist_yaw child link: {_short(wrist)}")
        if camera_edge is None:
            print("  camera fixed joint : missing")
        else:
            print(
                "  camera fixed joint : "
                f"{_short(camera_edge.body0)} active={camera_edge.body0_active} "
                f"--{camera_edge.joint_name}--> {_short(camera_edge.body1)} active={camera_edge.body1_active}"
            )
        if hand_edge is None:
            print("  hand fixed joint   : missing")
        else:
            print(
                "  hand fixed joint   : "
                f"{_short(hand_edge.body0)} active={hand_edge.body0_active} "
                f"--{hand_edge.joint_name}--> {_short(hand_edge.body1)} active={hand_edge.body1_active}"
            )

        is_accessory = (
            wrist is not None
            and camera_edge is not None
            and hand_edge is not None
            and camera_edge.joint_type == "PhysicsFixedJoint"
            and hand_edge.joint_type == "PhysicsFixedJoint"
            and camera_edge.body0 == wrist
            and hand_edge.body0 == wrist
            and camera_edge.body1 != hand_edge.body1
        )
        is_camera_removed_with_hand_kept = wrist is not None and camera_edge is None and hand_edge is not None
        if is_accessory:
            assessment = "separate fixed accessory"
        elif is_camera_removed_with_hand_kept:
            assessment = "camera accessory inactive/removed; hand remains attached"
        else:
            assessment = "check manually"
        print(f"  assessment         : {assessment}")


def _set_hand_camera_base_links_inactive(stage) -> None:
    """Deactivate hand camera base link prims in memory only."""

    print("[INFO] Deactivating hand camera base link prims in memory only.")
    for side in ("left", "right"):
        prim_path = f"/g1_29dof_with_hand_rev_1_0/{side}_hand_camera_base_link"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Missing prim: {prim_path}")
            continue
        print(f"[INFO] {prim_path}: active {prim.IsActive()} -> False")
        prim.SetActive(False)


def _set_hand_camera_base_joints_inactive(stage) -> None:
    """Deactivate hand camera base fixed joint prims in memory only."""

    print("[INFO] Deactivating hand camera base joint prims in memory only.")
    for side in ("left", "right"):
        prim_path = f"/g1_29dof_with_hand_rev_1_0/joints/{side}_hand_camera_base_joint"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Missing prim: {prim_path}")
            continue
        print(f"[INFO] {prim_path}: active {prim.IsActive()} -> False")
        prim.SetActive(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect USD PhysicsJoint parent-child relationships.")
    parser.add_argument("--usd_path", type=Path, default=DEFAULT_USD_PATH, help="USD file to inspect.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Regex used to filter printed joints.")
    parser.add_argument("--all", action="store_true", help="Print every USD joint edge.")
    parser.add_argument(
        "--body",
        action="append",
        default=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        help="Print incoming/outgoing joints for a body name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no_hand_camera_check",
        action="store_true",
        help="Skip the specialized hand camera accessory assessment.",
    )
    parser.add_argument(
        "--deactivate_hand_camera_base_links",
        action="store_true",
        help="Set left/right hand_camera_base_link prims inactive in memory before inspecting. The USD file is not saved.",
    )
    parser.add_argument(
        "--deactivate_hand_camera_base_joints",
        action="store_true",
        help="Set left/right hand_camera_base_joint prims inactive in memory before inspecting. The USD file is not saved.",
    )
    args = parser.parse_args()

    usd_path = _resolve_path(args.usd_path)
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    Usd = _load_usd_module()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    if args.deactivate_hand_camera_base_links:
        _set_hand_camera_base_links_inactive(stage)
    if args.deactivate_hand_camera_base_joints:
        _set_hand_camera_base_joints_inactive(stage)

    print(f"[INFO] USD: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    print(f"[INFO] defaultPrim: {default_prim.GetPath() if default_prim else None}")
    print()

    edges = _joint_edges(stage)
    _print_filtered_edges(edges, args.pattern, args.all)

    for body_name in args.body:
        _print_body_branch(edges, body_name)

    if not args.no_hand_camera_check:
        _print_hand_camera_assessment(edges)


if __name__ == "__main__":
    main()
