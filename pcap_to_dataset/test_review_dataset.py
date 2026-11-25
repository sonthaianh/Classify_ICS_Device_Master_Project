"""Semantic regression tests for packet windows, evidence and ML separation."""
import importlib.util
from collections import defaultdict
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_contract import TRAINING_FEATURES, select_training_features

spec = importlib.util.spec_from_file_location("review_builder", Path(__file__).with_name("23_build_review_dataset.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

A = "00:11:22:33:44:55"
B = "00:22:33:44:55:66"


def packet(ts, idx, reverse=False, **overrides):
    r = dict(timestamp=float(ts), src_mac=B if reverse else A, dst_mac=A if reverse else B,
             eth_type=0x0800, src_ip="10.0.0.2" if reverse else "10.0.0.1",
             dst_ip="10.0.0.1" if reverse else "10.0.0.2", ip_proto=6,
             src_port=102 if reverse else 50000, dst_port=50000 if reverse else 102,
             tcp_flags=0x12 if reverse else 0x02, packet_len=100, app_payload=b"", packet_idx=idx, vlan_id=None)
    r.update(overrides)
    return r


class ReviewDatasetTests(unittest.TestCase):
    def run_build(self, rows, batch=2):
        registry = pd.DataFrame([dict(device_id="A", mac=A, vendor="test", ground_truth_family="OT"),
                                 dict(device_id="B", mac=B, vendor="test", ground_truth_family="IT")]).set_index("device_id")
        with tempfile.TemporaryDirectory() as folder:
            # build_capture expects files under ROOT for traceable relative paths.
            path = Path(folder) / "capture.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            with patch.object(builder, "ROOT", Path(folder)), patch.object(builder, "decode_samples", return_value={"status": "test_no_dissection", "frames": 0}):
                return builder.build_capture(path, {A: "A", B: "B"}, registry, {102}, "unused", [], batch_size=batch)

    def test_packet_windows_no_uniform_allocation_or_future_duration(self):
        windows, review, _ = self.run_build([packet(1, 0), packet(2, 1, True), packet(100, 2, tcp_flags=0x10)])
        a = windows[windows.device_id == "A"].set_index("window_start")
        self.assertEqual(a.loc[0, "pkt_alloc"], 2)
        self.assertEqual(a.loc[90, "pkt_alloc"], 1)
        self.assertNotIn(45, a.index)  # no invented packets in a silent window
        self.assertEqual(a.loc[0, "flow_dur_median"], 1)
        self.assertEqual(a.loc[90, "flow_dur_median"], 0)
        self.assertEqual(a.loc[90, "carryover_flow_ratio"], 1)
        self.assertTrue(review.label.eq("").all())
        self.assertTrue(review.review_status.eq("pending").all())

    def test_duplicate_across_batches_and_batch_size_invariance(self):
        first = packet(1, 0)
        duplicate = {**first, "packet_idx": 1}
        rows = [first, duplicate, packet(2, 2, True)]
        w1, _, audit = self.run_build(rows, 1)
        w2, _, _ = self.run_build(rows, 10)
        pd.testing.assert_frame_equal(w1.sort_values(builder.KEY).reset_index(drop=True), w2.sort_values(builder.KEY).reset_index(drop=True))
        self.assertEqual(audit["duplicate_flow_packets_excluded"], 1)
        self.assertEqual(w1[w1.device_id == "A"].iloc[0].pkt_alloc, 2)
        self.assertEqual(w1[w1.device_id == "A"].iloc[0].packet_count, 3)

    def test_packet_only_devices_not_lost_and_unparsed_ip_not_nonip(self):
        rows = [packet(1, 0, eth_type=0x0806, src_ip=None, dst_ip=None, ip_proto=None,
                       src_port=None, dst_port=None),
                packet(2, 1, eth_type=0x86DD, src_ip=None, dst_ip=None, ip_proto=None,
                       src_port=None, dst_port=None)]
        windows, _, _ = self.run_build(rows)
        a = windows[windows.device_id == "A"].iloc[0]
        self.assertFalse(a.has_flow_packets)
        self.assertTrue(np.isnan(a.pkts_per_flow))
        self.assertEqual(a.nonip_frame_count, 1)
        self.assertEqual(a.unparsed_ip_frame_count, 1)

    def test_multicast_is_evidence_not_device_and_arp_target(self):
        arp = b"\x00\x01\x08\x00\x06\x04\x00\x01" + bytes(16) + bytes([10, 0, 0, 9])
        windows, review, _ = self.run_build([packet(1, 0, dst_mac="ff:ff:ff:ff:ff:ff", eth_type=0x0806,
                     ip_proto=None, src_port=None, dst_port=None, src_ip=None, dst_ip=None, app_payload=arp)])
        self.assertEqual(windows.device_id.tolist(), ["A"])
        self.assertEqual(windows.iloc[0].arp_target_unique, 1)
        self.assertEqual(windows.iloc[0].bcast_frame_ratio, 1)

    def test_later_handshake_does_not_change_earlier_role(self):
        # A initially sends from service port: heuristic server. Later SYN is
        # in the opposite direction, and must not rewrite the earlier window.
        rows = [packet(1, 0, src_port=102, dst_port=50000, tcp_flags=0x10),
                packet(100, 1, src_port=102, dst_port=50000, tcp_flags=0x02)]
        w, _, _ = self.run_build(rows)
        a = w[w.device_id == "A"].set_index("window_start")
        self.assertEqual(a.loc[0, "server_role_ratio"], 1)
        self.assertEqual(a.loc[0, "role_syn_ratio"], 0)
        self.assertEqual(a.loc[90, "server_role_ratio"], 0)

    def test_training_allowlist_rejects_numeric_metadata(self):
        frame = pd.DataFrame({c: [1.0] for c in TRAINING_FEATURES} | {"device_id": [123], "ot_pkt_share": [0.9], "label": [1]})
        self.assertEqual(select_training_features(frame).columns.tolist(), TRAINING_FEATURES)
        with self.assertRaises(ValueError):
            select_training_features(frame, ["ot_pkt_share"])

    def test_training_without_reviewed_labels_stops_before_fit(self):
        script = Path(__file__).with_name("17_train_baseline_model.py")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Labeling is deferred", result.stderr)

    def test_unknown_is_never_silently_negative(self):
        script = Path(__file__).with_name("17_train_baseline_model.py")
        result = subprocess.run([sys.executable, str(script), "--negative-label", "unknown"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown cannot be treated as non-OT", result.stderr)

    def test_decoded_identity_is_not_copied_to_both_endpoints(self):
        row = packet(1, 7, app_payload=b"example")
        row["_owners"], row["_sender_owners"] = ["A", "B"], ["A"]
        fields = ["frame.number", "frame.protocols", "s7comm.szl.001c.0007.cputypname"]
        result = subprocess.CompletedProcess([], 0, stdout=b'1\teth:ip:tcp:s7comm\tSIMATIC S7-1200\n', stderr=b"")
        evidence = defaultdict(builder.empty_evidence)
        with patch.object(builder.subprocess, "run", return_value=result):
            builder.decode_samples([row], "unused", fields, evidence)
        self.assertEqual(evidence["A"]["models"], {"SIMATIC S7-1200"})
        self.assertEqual(evidence["B"]["models"], set())
        self.assertIn("s7comm", evidence["B"]["protocols"])

    def test_unicode_capture_name_does_not_crash_windows_console(self):
        folder = str(Path(__file__).resolve().parent)
        code = f"import sys; sys.path.insert(0, {folder!r}); import feature_contract; print(chr(0x131))"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                env={**os.environ, "PYTHONIOENCODING": "cp1252"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.decode("utf-8").strip(), "\u0131")


if __name__ == "__main__":
    unittest.main()
