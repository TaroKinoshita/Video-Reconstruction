"""Venue layout (docs/VENUE_PLAN.md phases C-D): a structured description of a whole venue, the parts it is made of,
and the step that turns it into scene objects fitted to what the source view showed.

    layout.py    the layout format (parts: court, net, ground, stand, wall, box, canopy) and its checks
    generate.py  meshes for the parts from shared operations (rows along a front line, boxes along a line)
    fit.py       moving a layout's numbers onto the observations (analysis/observations.py)
    build.py     layout + observed things -> Project
"""
