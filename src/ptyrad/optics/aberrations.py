from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

# ------------------------------------------------------------
# Specification & Constants
# ------------------------------------------------------------

ABERRATION_SPEC = {
    (1, 0): ("C1", 1.0, "Defocus (C10 = -df)"),
    (1, 2): ("A1", 1.0, "2-fold astigmatism"),
    (2, 1): ("B2", 1/3.0, "Axial coma"),
    (2, 3): ("A2", 1.0, "3-fold astigmatism"),
    (3, 0): ("C3", 1.0, "Spherical aberration"),
    (3, 2): ("S3", 1/4.0, "Axial star aberration"),
    (3, 4): ("A3", 1.0, "4-fold astigmatism"),
    (4, 1): ("B4", 1/4.0, "Axial coma(4th)"),
    (4, 3): ("D4", 1/4.0, "3-lobe aberration"),
    (4, 5): ("A4", 1.0, "5-fold astigmatism"),
    (5, 0): ("C5", 1.0, "Spherical aberration (5th)"),
    (5, 2): ("S5", 1/6.0, "Axial star aberration(5th)"),
    (5, 4): ("R5", 1/6.0, "4-lobe aberration"),
    (5, 6): ("A5", 1.0, "6-fold astigmatism"),
}

KRIVANEK_TO_HAIDER = {nm: (h, s) for nm, (h, s, _) in ABERRATION_SPEC.items()}
HAIDER_TO_KRIVANEK = {}
for nm, (h, s, _) in ABERRATION_SPEC.items():
    HAIDER_TO_KRIVANEK[h] = (1/s, nm)
    if nm[1] > 0:
        HAIDER_TO_KRIVANEK[f"{h}phi"] = (1.0, nm) # Angle doesn't need scaling

# Aliases: (target_key, scale_factor)
# defocus convention: positive defocus = underfocus = negative C10
ALIASES = {
    "defocus": ("C10", -1.0),
    "Cs": ("C30", 1.0),
}

# ------------------------------------------------------------
# Internal Structures
# ------------------------------------------------------------

@dataclass(frozen=True)
class ParsedKey:
    """The resolved address and context for a user key."""
    nm: Tuple[int, int]
    param: Literal["magnitude", "angle", "a", "b"]
    scale: float

@dataclass(frozen=True)
class AberrationValue:
    """Intermediate storage for export."""
    nm: Tuple[int, int]
    mag: float
    angle: float

# ------------------------------------------------------------
# Main Class
# ------------------------------------------------------------

class Aberrations:
    """
    Handles probe aberrations with support for Krivanek/Haider notations.
    Internal state is always Polar Krivanek (Magnitude [Å], Angle [deg]).
    
    Note that for round lens aberrations (m=0), 
    the internal storage is always (mag, 0.0) for type uniformity.
    However, setting or getting "components" from these round lens aberrations
    is strictly forbidden for users. For example, 'phi30', 'C10a' are not allowed.
    """

    def __init__(self, data: Optional[Dict] = None):
        self._data: Dict[Tuple[int, int], Tuple[float, float]] = {}
        if data is not None:
            self._parse_and_normalize(data)

    # ============================================================
    # Public API
    # ============================================================

    def __getitem__(self, key: str) -> float:
        pk = self._parse_external_key(key)
        
        if pk.nm not in self._data:
            raise KeyError(f"Aberration {key} (Order {pk.nm}) not set.")

        mag, ang = self._data[pk.nm]
        
        # Apply Inverse Scale (User Units = Internal / Scale)
        scale = 1.0 / pk.scale

        if pk.param == "magnitude":
            return mag * scale

        if pk.param == "angle":
            return ang # Angles are not scaled

        if pk.param in ["a", "b"]:
            a, b = self._polar_to_cartesian(mag, ang, pk.nm[1])
            return (a if pk.param == "a" else b) * scale

    def __setitem__(self, key: str, value: float):
        pk = self._parse_external_key(key)
        value = float(value) * pk.scale
        
        # --- Path 1: Round Aberrations (m=0) ---
        if pk.nm[1] == 0:
            # This angle=0 is only for internal representation
            # Setting 'phi10' or 'C10a' will be blocked by _parse_external_key
            # Exporting m=0 aberrations would also only have a single value across 'polar', 'complex', 'cartesian'.
            self._data[pk.nm] = (value, 0.0) 
            return

        # --- Path 2: Non-Round Aberrations (m > 0) ---
        curr_mag, curr_ang = self._data.get(pk.nm, (0.0, 0.0))

        if pk.param == "magnitude":
            self._data[pk.nm] = (value, curr_ang)
            return

        if pk.param == "angle":
            self._data[pk.nm] = (curr_mag, value)
            return

        if pk.param in ["a", "b"]:
            a, b = self._polar_to_cartesian(curr_mag, curr_ang, pk.nm[1])

            if pk.param == "a":
                self._data[pk.nm] = self._cartesian_to_polar(value, b, pk.nm[1])
            else:
                self._data[pk.nm] = self._cartesian_to_polar(a, value, pk.nm[1])
            return
        
    def get_coefficients(self, style: Literal["polar", "cartesian", "complex"] = "cartesian") -> Dict[Tuple[int, int], Any]:
        """Get raw coefficients for computation (nested dictionary)."""
        return self.export(notation="krivanek", style=style, layout="nested", round_decimals=None)

    def get_haider(self, decimals=3) -> Dict[str, float]:
        """Get flattened dictionary in Haider notation."""
        return self.export(notation="haider", style="polar", layout="flat", round_decimals=decimals)
    
    def get_krivanek_polar(self, decimals=3) -> Dict[str, float]:
        """Export Krivanek Polar (Flat dictionary)."""
        return self.export(notation='krivanek', style='polar', layout='flat', round_decimals=decimals)

    def get_krivanek_cartesian(self, decimals=3) -> Dict[str, float]:
        """Export Krivanek Cartesian (Flat dictionary)."""
        return self.export(notation='krivanek', style='cartesian', layout='flat', round_decimals=decimals)
    
    # ============================================================
    # Export Engine
    # ============================================================

    def export(self, 
               notation: Literal["krivanek", "haider"] = "krivanek",
               style: Literal["polar", "cartesian", "complex"] = "polar",
               layout: Literal["flat", "nested"] = "flat",
               round_decimals: Optional[int] = 3) -> Dict:
        
        values = self._collect_values()
        result = {}

        for v in values:
            # 1. Map & Scale
            name, mag, ang, m = self._map_notation(v, notation)
            if name is None:
                continue # Skip if not in notation

            # 2. Format Value
            payload = self._format_style(mag, ang, m, style, round_decimals)

            # 3. Apply Layout
            self._apply_layout(result, v.nm, name, payload, notation, style, layout)

        return result

    def _collect_values(self):
        """ Return a sorted list of aberration values """
        return [AberrationValue(nm, mag, ang) for nm, (mag, ang) in sorted(self._data.items())]

    def _map_notation(self, v: AberrationValue, notation: str):
        """ Convert the names and values between Haider and Krivanek notations """
        n, m = v.nm
        if notation == "haider":
            if v.nm not in KRIVANEK_TO_HAIDER:
                return None, None, None, None
            name, scale = KRIVANEK_TO_HAIDER[v.nm]
            return name, v.mag * scale, v.angle, m
        
        # Default Krivanek
        return f"C{n}{m}", v.mag, v.angle, m

    def _format_style(self, mag, ang, m, style, decimals):
        """ Convert aberration values to 'polar', 'cartesian', and 'complex' format """
        if style == "complex":
            rad = np.radians(ang)
            return mag * np.exp(1j * m * rad)

        if style == "cartesian":
            a, b = self._polar_to_cartesian(mag, ang, m)
            if decimals is not None:
                a, b = round(a, decimals), round(b, decimals)
            
            if m == 0:
                return a if decimals is None else round(a, decimals)
            return {"a": a, "b": b}

        # Polar
        if decimals is not None:
            mag, ang = round(mag, decimals), round(ang, decimals)
        
        if m == 0:
            return mag
        return {"mag": mag, "phi": ang}

    def _apply_layout(self, result, nm, base, payload, notation, style, layout):
        """ Construct output dict as 'nested' or 'flat' layout """
        if layout == "nested":
            result[nm] = payload
            return

        # Flat Layout
        if not isinstance(payload, dict):
            # Scalar payload (m=0 or complex)
            result[base] = self._to_native(payload)
            return

        for k, v in payload.items():
            # Suffix logic
            if style == "cartesian":
                key = f"{base}{k}"
            else: # polar
                if k == "mag":
                    key = base
                else: # phi
                    key = f"{base}phi" if notation == "haider" else f"phi{nm[0]}{nm[1]}"
            
            result[key] = self._to_native(v)

    # ============================================================
    # Parsing Logic
    # ============================================================

    def _parse_and_normalize(self, raw: Dict):
        store = defaultdict(dict) # Temporary dict to stage the intermediate items

        for key, val in raw.items():
            # Tuple Keys (Canonical)
            if isinstance(key, tuple):
                n, m = key
                if not self._is_valid_nm(n, m):
                    raise ValueError(f"Invalid order {key}")
                self._parse_tuple_val(store, key, val)
                continue

            # String Keys (User Input)
            pk = self._parse_external_key(key)
            val = float(val) * pk.scale
            self._assign(store, pk.nm, pk.param, val)

        # Finalize (Store -> Internal State)
        for nm, params in store.items():
            self._finalize_term(nm, params)

    def _finalize_term(self, nm, params):
        m = nm[1]
        
        # Conflict Check
        has_polar = "magnitude" in params or "angle" in params
        has_cart = "a" in params or "b" in params
        if has_polar and has_cart:
            raise ValueError(f"Conflicting notation for term {nm}")

        if m == 0:
            mag = params.get("magnitude", params.get("a", 0.0))
            if mag != 0:
                self._data[nm] = (mag, 0.0)
            return

        if has_polar:
            mag = params.get("magnitude", 0.0)
            ang = params.get("angle", 0.0)
            if mag != 0:
                self._data[nm] = (mag, ang)
        else:
            a = params.get("a", 0.0)
            b = params.get("b", 0.0)
            mag, ang = self._cartesian_to_polar(a, b, m)
            if mag != 0:
                self._data[nm] = (mag, ang)

    def _parse_external_key(self, key: str) -> ParsedKey:
        scale = 1.0
        raw = key

        # Resolve Alias/Haider
        if raw in ALIASES:
            raw, s = ALIASES[raw]
            scale *= s
        elif raw in HAIDER_TO_KRIVANEK:
            s, nm = HAIDER_TO_KRIVANEK[raw]
            if raw.endswith("phi"): 
                # Strict check: Haider angle aliases (if any exist for m=0)
                if nm[1] == 0:
                     raise ValueError(f"Invalid key '{key}': Round aberration {nm} has no angle.")
                return ParsedKey(nm, "angle", scale)
            return ParsedKey(nm, "magnitude", scale * s)

        # Parse Standard Keys
        if raw.startswith("phi"):
            nm = self._parse_nm(raw[3:])
            if nm[1] == 0:
                raise ValueError(f"Invalid key '{key}': Round aberration C{nm[0]}0 has no angle.")
            return ParsedKey(nm, "angle", scale)
        
        if raw.startswith("C"):
            body = raw[1:]
            
            # 1. Identify Component
            if body.endswith("a"):
                param = "a"
                nm_str = body[:-1]
            elif body.endswith("b"):
                param = "b"
                nm_str = body[:-1]
            else:
                param = "magnitude"
                nm_str = body

            # 2. Parse Order
            nm = self._parse_nm(nm_str)

            # 3. STRICT VALIDATION for Round Lenses (m=0)
            if nm[1] == 0:
                if param in ("a", "b"):
                    raise ValueError(f"Invalid key '{key}': Round aberration C{nm[0]}0 is a scalar. Use 'C{nm[0]}0', not 'C{nm[0]}0{param}'.")
            
            return ParsedKey(nm, param, scale)

        raise KeyError(f"Unknown key: {key}")

    def _parse_nm(self, s: str) -> Tuple[int, int]:
        if len(s) >= 2 and s.isdigit():
            n, m = int(s[0]), int(s[1:])
            if self._is_valid_nm(n, m):
                return (n, m)
        raise KeyError(f"Invalid order string: {s}")

    def _parse_tuple_val(self, store, nm, val):
        if isinstance(val, complex):
            self._assign(store, nm, "a", val.real)
            self._assign(store, nm, "b", val.imag)
        elif isinstance(val, dict):
            for k, v in val.items():
                target = None
                if k in ("a", "b"):
                    target = k
                elif k in ("mag", "magnitude"):
                    target = "magnitude"
                elif k in ("phi", "angle"):
                    target = "angle"
                if target:
                    self._assign(store, nm, target, v)
        elif isinstance(val, (int, float, np.number)):
            if nm[1] != 0:
                raise ValueError(f"Ambiguous scalar for non-round term {nm}")
            self._assign(store, nm, "magnitude", float(val))
        else:
            raise TypeError(f"Invalid value type for {nm}: {type(val)}")

    def _assign(self, store, nm, param, val):
        if param in store[nm]:
            raise ValueError(f"Conflicting value for {nm} parameter '{param}'")
        store[nm][param] = float(val)

    # ============================================================
    # Helpers
    # ============================================================

    def _polar_to_cartesian(self, mag, ang, m):
        r = np.radians(ang)
        return mag * np.cos(m * r), mag * np.sin(m * r)

    def _cartesian_to_polar(self, a, b, m):
        mag = np.sqrt(a * a + b * b)
        if mag == 0:
            return 0.0, 0.0
        return mag, np.degrees(np.arctan2(b, a) / m)

    @staticmethod
    def _is_valid_nm(n: int, m: int) -> bool:
        if n < 1 or m < 0 or m > n + 1:
            return False
        return (m % 2) == (0 if n % 2 == 1 else 1)

    @staticmethod
    def _to_native(val):
        if isinstance(val, np.number):
            return val.item()
        return val

    # ============================================================
    # Human Readability
    # ============================================================

    def pretty_print(self) -> str:
        """
        Generate a formatted table of current aberrations.
        """
        if not self._data:
            return "Aberrations(Empty)"

        rows = []
        max_mag_len = 9  # Start with min width for "Magnitude" header
        
        for (n, m), (mag, ang) in sorted(self._data.items()):
            # 1. Krivanek Label
            kriv_label = f"C{n},{m}"
            
            # 2. Haider Label (e.g. "3*B2")
            haider_label = "-"
            desc = ""
            
            if (n, m) in ABERRATION_SPEC:
                code, scale, name = ABERRATION_SPEC[(n, m)]
                desc = name
                
                # Format: "3*B2" if scale is 1/3, "B2" if scale is 1
                if scale == 1.0:
                    haider_label = code
                else:
                    # Calculate inverse factor (e.g. 1 / 0.3333 = 3.0)
                    inv_scale = 1.0 / scale
                    
                    # Check if integer (with tolerance for float math)
                    if abs(inv_scale - round(inv_scale)) < 1e-5:
                        factor = int(round(inv_scale))
                        haider_label = f"{factor}*{code}"
                    else:
                        # Fallback for non-integers (e.g. 1.5*C1)
                        haider_label = f"{inv_scale:.2g}*{code}"

            # 3. Format Magnitude (Track max width)
            mag_str = f"{mag:.4f}" 
            max_mag_len = max(max_mag_len, len(mag_str))

            # 4. Format Angle
            if m == 0:
                ang_str = "-"
            else:
                ang_str = f"{ang:.2f}"

            rows.append((kriv_label, haider_label, mag_str, ang_str, desc))

        # --- Table Construction ---
        
        # Dynamic padding for Magnitude column
        col_mag_width = max_mag_len + 2
        
        # Format: Krivanek | Haider | Mag | Angle | Desc
        # Fixed widths for labels, dynamic for values
        row_fmt = f"{{:<8}} {{:<10}} {{:<{col_mag_width}}} {{:<10}} {{}}"
        
        header = row_fmt.format("Krivanek", "Haider", "Magnitude", "Angle (°)", "Description")
        separator = "-" * (8 + 10 + col_mag_width + 10 + 40) # Rough total length

        lines = [header, separator]
        for r in rows:
            lines.append(row_fmt.format(*r))

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the pretty-printed table for print(model)."""
        return self.pretty_print()
    
    def __repr__(self) -> str:
        """Concise debug representation."""
        return f"<Aberrations: {len(self._data)} terms set>"