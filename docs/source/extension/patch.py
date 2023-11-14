#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import List, Optional

import sphinx.ext.autosummary.generate as autosummary


def monkey_patch_find_autosummary_in_lines(
    lines: List[str],
    module: str = None,
    filename: str = None,
) -> autosummary.AutosummaryEntry:
    # Monkey-patches `sphinx.ext.autosummary.find_autosummary_in_lines`
    # to parse for-loops and import their content.
    # Most of its implementation is directly copied over from `sphinx`, see:
    # https://github.com/sphinx-doc/sphinx/sphinx/ext/autosummary/generate.py

    import importlib
    import os.path as osp
    import re

    autosummary_re = re.compile(r"^(\s*)\.\.\s+autosummary::\s*")
    automodule_re = re.compile(r"^\s*\.\.\s+automodule::\s*([A-Za-z0-9_.]+)\s*$")
    module_re = re.compile(r"^\s*\.\.\s+(current)?module::\s*([a-zA-Z0-9_.]+)\s*$")
    autosummary_item_re = re.compile(r"^\s+(~?[_a-zA-Z][a-zA-Z0-9_.]*)\s*.*?")
    recursive_arg_re = re.compile(r"^\s+:recursive:\s*$")
    toctree_arg_re = re.compile(r"^\s+:toctree:\s*(.*?)\s*$")
    template_arg_re = re.compile(r"^\s+:template:\s*(.*?)\s*$")
    list_arg_re = re.compile(r"^\s+{% for\s*(.*?)\s*in\s*(.*?)\s*%}$")

    documented: list[autosummary.AutosummaryEntry] = []

    recursive = False
    toctree: Optional[str] = None
    template = None
    curr_module = module
    in_autosummary = False
    base_indent = ""

    for line in lines:
        if in_autosummary:
            m = recursive_arg_re.match(line)
            if m:
                recursive = True
                continue

            m = toctree_arg_re.match(line)
            if m:
                toctree = m.group(1)
                if filename:
                    toctree = osp.join(osp.dirname(filename), toctree)
                continue

            m = template_arg_re.match(line)
            if m:
                template = m.group(1).strip()
                continue

            # Begin of modified part #####################
            m = list_arg_re.match(line)
            if m:
                obj_name = m.group(2).strip()
                module_name, obj_name = obj_name.rsplit(".", maxsplit=1)
                module = importlib.import_module(module_name)
                for entry in getattr(module, obj_name):
                    documented.append(
                        autosummary.AutosummaryEntry(
                            f"{module_name}.{entry}",
                            toctree,
                            template,
                            recursive,
                        )
                    )
                continue
            # End of modified part ######################

            if line.strip().startswith(":"):
                continue

            m = autosummary_item_re.match(line)
            if m:
                name = m.group(1).strip()
                if name.startswith("~"):
                    name = name[1:]
                if curr_module and not name.startswith(f"{curr_module}."):
                    name = f"{curr_module}.{name}"
                documented.append(
                    autosummary.AutosummaryEntry(
                        name,
                        toctree,
                        template,
                        recursive,
                    )
                )
                continue

            if not line.strip() or line.startswith(f"{base_indent} "):
                continue

            in_autosummary = False

        m = autosummary_re.match(line)
        if m:
            in_autosummary = True
            base_indent = m.group(1)
            recursive = False
            toctree = None
            template = None
            continue

        m = automodule_re.search(line)
        if m:
            curr_module = m.group(1).strip()
            # recurse into the automodule docstring
            documented.extend(
                autosummary.find_autosummary_in_docstring(
                    curr_module,
                    filename=filename,
                )
            )
            continue

        m = module_re.match(line)
        if m:
            curr_module = m.group(2)
            continue

    return documented


def setup(app):
    # Monkey-patch `sphinx.ext.autosummary.find_autosummary_in_lines`:
    autosummary.find_autosummary_in_lines = monkey_patch_find_autosummary_in_lines

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
