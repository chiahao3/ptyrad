{{ fullname | escape | underline}}

{% if modules %}

{# =========================================================================
   CASE 1: PACKAGE (e.g. ptyrad.params, ptyrad.utils)
   Goal: Show ONLY the table of sub-modules. No classes, no noise.
   ========================================================================= #}

.. automodule:: {{ fullname }}
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}

{# 
   CRITICAL: We intentionally deleted the 'Classes' and 'Functions' blocks here.
   This prevents PtyRADParams or InitParams from cluttering the landing page.
   You MUST click into 'init_params' to see them.
#}

{% else %}

{# =========================================================================
   CASE 2: MODULE (e.g. ptyrad.params.init_params)
   Goal: Table at top -> Details at bottom.
   ========================================================================= #}

.. automodule:: {{ fullname }}
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

{# --- Summary Tables --- #}

{% block classes_leaf %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions_leaf %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{# --- Full Details --- #}

.. rubric:: Details

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

{% endif %}