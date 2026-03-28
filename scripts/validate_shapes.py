#!/usr/bin/env python3
"""
Validate tensor shapes in inference and model code.
Checks for common shape mismatch patterns that caused bugs in prior sessions.
"""

import sys
import re
from pathlib import Path


def check_tensor_shapes(filepath):
    """Check file for tensor shape validation patterns."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        return True, f"Could not read file: {e}"

    issues = []

    # Check if this is an inference/model file
    is_inference = any(x in filepath.lower() for x in ['inference', 'generate', 'sample'])

    if is_inference:
        # For inference files, check for shape assertions
        has_shape_assertions = 'assert' in content and 'shape' in content
        has_tensor_ops = 'torch.' in content or 'tensor' in content.lower()

        if has_tensor_ops and not has_shape_assertions:
            issues.append("⚠ Inference file has tensor ops but no shape assertions")

        # Check for reshape/view without documentation
        if re.search(r'\.reshape\(|\.view\(', content):
            # Count reshape operations
            reshape_count = len(re.findall(r'\.reshape\(|\.view\(', content))
            # Check if each has a comment or assertion
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '.reshape(' in line or '.view(' in line:
                    has_context = False
                    # Check previous or next line for shape info
                    if i > 0 and ('shape' in lines[i-1].lower() or '#' in lines[i-1]):
                        has_context = True
                    if i < len(lines)-1 and ('shape' in lines[i+1].lower() or '#' in lines[i+1]):
                        has_context = True
                    if not has_context:
                        issues.append(f"⚠ Line {i+1}: reshape/view without shape context")
                        break

    if issues:
        return False, "\n".join(issues)
    return True, "✓ Shape validation passed"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("✓ No file specified")
        sys.exit(0)

    filepath = sys.argv[1]

    # Only check Python files
    if not filepath.endswith('.py'):
        sys.exit(0)

    # Only validate inference/model files
    relevant = any(x in filepath.lower() for x in ['inference', 'generate', 'sample', 'model', 'decoder'])
    if not relevant:
        sys.exit(0)

    success, message = check_tensor_shapes(filepath)
    print(message)
    sys.exit(0 if success else 1)
