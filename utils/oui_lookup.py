#!/usr/bin/env python3
r"""
oui_lookup.py

Tra cuu vendor phan cung tu dia chi MAC, dung truc tiep file
Tools/utils/OUI/manuf.txt (IEEE OUI registry, xuat boi `tshark -G manuf`,
CHINH LA file Wireshark/tshark tu dung de resolve ten vendor trong GUI).

Day la TIN HIEU DOC LAP voi viec tai tao frame: src_mac/dst_mac trong
raw_parquet la gia tri THAT, khong bi anh huong boi gioi han cua
frame_builder.py (checksum/seq/VLAN tai tao). Vi vay vendor tu MAC OUI
dang tin cay ngay ca khi header L3/L4 la synthetic.

Ho tro dung dinh dang block IEEE: 24-bit (MA-L, mac dinh khi khong co "/N"),
28-bit (MA-M), 36-bit (MA-S) - vi du dong "00:1B:C5:00:00/36" trong manuf.txt.

Su dung:
    lookup = OuiLookup()  # tu dong tim manuf.txt canh file nay (Tools/utils/OUI/manuf.txt)
    info = lookup.lookup("00:0f:b5:4d:be:f3")
    # info = {"short_name": ..., "full_name": ..., "match_bits": 24,
    #         "is_multicast": False, "is_locally_administered": False}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_MANUF_PATH = Path(__file__).resolve().parent / "OUI" / "manuf.txt"


class OuiLookup:
    def __init__(self, manuf_path: Optional[Path] = None) -> None:
        self.manuf_path = Path(manuf_path) if manuf_path else DEFAULT_MANUF_PATH
        # by_bitlen[bitlen][masked_prefix_int] = (short_name, full_name)
        self._by_bitlen: Dict[int, Dict[int, Tuple[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.manuf_path.is_file():
            raise FileNotFoundError(f"Khong tim thay manuf.txt tai {self.manuf_path}")

        with open(self.manuf_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                prefix_field = parts[0].strip()
                short_name = parts[1].strip()
                full_name = parts[2].strip() if len(parts) > 2 else short_name

                parsed = self._parse_prefix(prefix_field)
                if parsed is None:
                    continue
                bitlen, masked_value = parsed
                self._by_bitlen.setdefault(bitlen, {})[masked_value] = (short_name, full_name)

        self._sorted_bitlens = sorted(self._by_bitlen.keys(), reverse=True)

    @staticmethod
    def _parse_prefix(prefix_field: str) -> Optional[Tuple[int, int]]:
        if "/" in prefix_field:
            hex_part, bits_part = prefix_field.split("/", 1)
            try:
                bitlen = int(bits_part)
            except ValueError:
                return None
        else:
            hex_part = prefix_field
            bitlen = None

        hex_digits = hex_part.replace(":", "").replace("-", "").strip()
        if not hex_digits:
            return None
        if bitlen is None:
            bitlen = len(hex_digits) * 4
        bitlen = max(1, min(bitlen, 48))

        # pad/truncate to 12 hex chars (48 bit), missing digits = 0
        hex_digits = (hex_digits + "0" * 12)[:12]
        try:
            value = int(hex_digits, 16)
        except ValueError:
            return None

        masked_value = value >> (48 - bitlen)
        return bitlen, masked_value

    @staticmethod
    def _mac_to_int(mac_str: str) -> Optional[int]:
        try:
            hex_digits = mac_str.replace(":", "").replace("-", "").strip()
            if len(hex_digits) != 12:
                return None
            return int(hex_digits, 16)
        except (ValueError, AttributeError):
            return None

    def lookup(self, mac_str: Optional[str]) -> Dict[str, object]:
        """Tra ve dict mo ta vendor cho 1 dia chi MAC. Luon co ket qua (khong
        bao gio None) - neu khong tim thay, short_name/full_name = None va
        ly do duoc phan anh qua is_multicast/is_locally_administered."""
        result = {
            "mac": mac_str,
            "short_name": None,
            "full_name": None,
            "match_bits": None,
            "is_multicast": False,
            "is_locally_administered": False,
        }
        mac_int = self._mac_to_int(mac_str) if mac_str else None
        if mac_int is None:
            return result

        first_octet = (mac_int >> 40) & 0xFF
        result["is_multicast"] = bool(first_octet & 0x01)
        result["is_locally_administered"] = bool(first_octet & 0x02)

        for bitlen in self._sorted_bitlens:
            masked = mac_int >> (48 - bitlen)
            hit = self._by_bitlen[bitlen].get(masked)
            if hit is not None:
                result["short_name"], result["full_name"] = hit
                result["match_bits"] = bitlen
                break

        return result

    def __len__(self) -> int:
        return sum(len(d) for d in self._by_bitlen.values())


if __name__ == "__main__":
    lookup = OuiLookup()
    print(f"Da nap {len(lookup)} OUI entries tu {lookup.manuf_path}")
    for test_mac in ["00:0f:b5:4d:be:f3", "00:1B:C5:00:25", "ff:ff:ff:ff:ff:ff", "02:00:00:00:00:01"]:
        print(test_mac, "->", lookup.lookup(test_mac))
