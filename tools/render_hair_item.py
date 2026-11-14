"""Blender: a hair item .glb on a VRM body with the body's own hair hidden, rendered from the front, the side and the back.

  BLENDER -b --python tools/render_hair_item.py -- BODY.vrm HAIR.glb OUT.png
Needs the VRM Add-on for Blender. A check of placement only (rest pose, flat lighting).
The add-on turns a VRM 0.x body to face Blender's front on import, while the glTF importer takes the item as it is
written: for a 0.x body give an item written in VRM 1.0 axes (`hair-item --body BODY.json`, not `--vrm`), or the
hair shows back to front here although it matches the file.
"""
import math
import sys

import bpy
from mathutils import Vector

body, hair, out = sys.argv[sys.argv.index('--') + 1:]
for o in list(bpy.data.objects):  # the start-up scene; factory settings would switch the VRM add-on off
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.import_scene.vrm(filepath=body)
hidden = []
for o in list(bpy.data.objects):
    if o.type == 'MESH' and ('hair' in o.name.lower() or 'hair' in o.data.name.lower()):
        o.hide_render = True
        hidden.append(o.name)
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=hair)
new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
for o in new:  # the importer's bone-shape helper is not hair
    if len(o.data.vertices) < 100 or o.parent is None:
        o.hide_render = True
new = [o for o in new if not o.hide_render]
lo = Vector((1e9,) * 3)
hi = Vector((-1e9,) * 3)
for o in new:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo = Vector(map(min, lo, w))
        hi = Vector(map(max, hi, w))
centre = (lo + hi) / 2
size = max(hi - lo)
print('HAIR_BBOX', [round(x, 4) for x in lo], [round(x, 4) for x in hi], 'hidden', hidden)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version < (5, 0, 0) else 'BLENDER_EEVEE'
scene.render.resolution_x = scene.render.resolution_y = 640
scene.render.film_transparent = False
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.85, 0.85, 0.85, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 1.2
scene.world = world
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = size * 1.6
tiles = []
for name, ang in (('front', -90), ('side', 0), ('back', 90)):
    a = math.radians(ang)
    cam.location = centre + Vector((math.cos(a) * 3, math.sin(a) * 3, 0))
    cam.rotation_euler = (math.radians(90), 0, a + math.radians(90))
    scene.render.filepath = out.replace('.png', f'_{name}.png')
    bpy.ops.render.render(write_still=True)
    tiles.append(scene.render.filepath)
print('RENDERED', tiles)
