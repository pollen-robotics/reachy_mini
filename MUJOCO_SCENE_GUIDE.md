# MuJoCo Scene Creation Guide for Reachy Mini

**Created:** October 18, 2025
**Purpose:** Learn how to create custom MuJoCo simulation environments for the Scene Manager Director Agent

---

## Quick Start

### Running the Desktop Viewer

```bash
# Activate virtual environment
source /Users/lauras/Desktop/laura/venv/bin/activate

# Load empty scene (just robot + floor)
python3 desktop_viewer.py --scene empty

# Load minimal scene (robot + table + objects)
python3 desktop_viewer.py --scene minimal

# Load your custom scene
python3 desktop_viewer.py --scene my_custom_scene
```

### Available Scenes

- **`empty`** - Minimal environment (robot + checkerboard floor)
- **`minimal`** - Interactive demo (robot + table + croissant + apple + rubber duck)

---

## MuJoCo Scene Anatomy

A scene is an XML file located at:
```
/Users/lauras/Desktop/laura/reachy_mini/src/reachy_mini/descriptions/reachy_mini/mjcf/scenes/
```

### Basic Structure

```xml
<mujoco model="scene">
    <!-- Include the robot model -->
    <include file="../reachy_mini.xml" />

    <!-- Set asset directories -->
    <compiler meshdir="../assets" texturedir="../assets" />

    <!-- Visual settings (camera, lighting, haze) -->
    <visual>...</visual>

    <!-- Assets (textures, materials, meshes) -->
    <asset>...</asset>

    <!-- The actual scene content -->
    <worldbody>...</worldbody>
</mujoco>
```

---

## Creating a Custom Scene

### Example: Simple Theater Stage

Let's create a scene for your director agent with a stage-like environment.

**File:** `src/reachy_mini/descriptions/reachy_mini/mjcf/scenes/theater_stage.xml`

```xml
<mujoco model="theater_stage">
    <include file="../reachy_mini.xml" />
    <compiler meshdir="../assets" texturedir="../assets" />

    <!-- Visual settings -->
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" offwidth="1280" offheight="720"/>
    </visual>

    <!-- Assets -->
    <asset>
        <!-- Sky gradient -->
        <texture type="skybox" builtin="gradient"
                 rgb1="0.1 0.05 0.2" rgb2="0 0 0"
                 width="512" height="3072" />

        <!-- Stage floor (dark wood texture) -->
        <texture type="2d" name="stage_floor" builtin="checker"
                 rgb1="0.3 0.2 0.1" rgb2="0.25 0.15 0.08"
                 markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="stage_material" texture="stage_floor"
                  texuniform="true" texrepeat="10 10" reflectance="0.1" />

        <!-- Spotlight effect (red curtain) -->
        <texture type="2d" name="curtain" builtin="flat"
                 rgb1="0.6 0.1 0.1" width="100" height="100" />
        <material name="curtain_material" texture="curtain" />
    </asset>

    <!-- Scene objects -->
    <worldbody>
        <!-- Overhead spotlight -->
        <light pos="0 0 2.5" dir="0 0 -1"
               directional="false"
               diffuse="1 0.9 0.7"
               specular="0.3 0.3 0.3"
               castshadow="true" />

        <!-- Stage floor (raised platform) -->
        <geom name="stage"
              type="box"
              size="2 2 0.1"
              pos="0 0 -0.1"
              material="stage_material" />

        <!-- Background curtain wall -->
        <geom name="curtain_back"
              type="box"
              size="3 0.05 2"
              pos="0 2 1"
              material="curtain_material" />
    </worldbody>
</mujoco>
```

**Run it:**
```bash
python3 desktop_viewer.py --scene theater_stage
```

---

## Key MuJoCo Concepts

### 1. Worldbody (Scene Objects)

The `<worldbody>` contains all physical objects in the scene.

#### Basic Shapes (Geoms)

```xml
<!-- Box -->
<geom name="box1" type="box" size="0.5 0.5 0.5" pos="1 0 0" />

<!-- Sphere -->
<geom name="ball" type="sphere" size="0.2" pos="0 1 0.5" />

<!-- Cylinder -->
<geom name="pillar" type="cylinder" size="0.1 1.0" pos="-1 0 0" />

<!-- Plane (infinite floor) -->
<geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" />
```

#### Dynamic Objects (with physics)

```xml
<body name="movable_cube" pos="0.5 0 0.5">
    <geom type="box" size="0.1 0.1 0.1" mass="0.5" />
    <joint name="cube_joint" type="free" />
</body>
```

- **`<body>`** - A rigid body that can move
- **`<joint type="free">`** - Allows 6-DOF movement (translation + rotation)
- **`mass`** - Makes it subject to gravity

### 2. Lighting

```xml
<!-- Directional light (like the sun) -->
<light pos="0 0 3" dir="0 0 -1" directional="true" />

<!-- Point light (like a bulb) -->
<light pos="1 1 2" directional="false"
       diffuse="1 0.8 0.6" castshadow="true" />
```

### 3. Materials & Textures

```xml
<!-- Built-in checker pattern -->
<texture type="2d" name="my_checker" builtin="checker"
         rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" />
<material name="my_mat" texture="my_checker" texrepeat="5 5" />

<!-- Solid color -->
<texture type="2d" name="red" builtin="flat" rgb1="0.8 0.1 0.1" />
<material name="red_mat" texture="red" />
```

### 4. Loading Custom Meshes

```xml
<!-- In <asset> section -->
<texture type="2d" name="my_texture" file="my_model/texture.png" />
<material name="my_material" texture="my_texture" />
<mesh name="my_mesh" file="my_model/model.obj" scale="1 1 1" />

<!-- In <worldbody> section -->
<body name="custom_object" pos="0 0 0.5">
    <geom type="mesh" mesh="my_mesh" material="my_material" />
    <joint type="free" />
</body>
```

**Mesh file location:**
```
src/reachy_mini/descriptions/reachy_mini/mjcf/assets/my_model/
```

---

## Director Agent Scene Ideas

### 1. Dual Agent Stage
Two robots facing each other for debates/conversations.

```xml
<worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" />
    <geom name="floor" type="plane" material="stage_material" />

    <!-- Position markers for two robots -->
    <geom name="position_a" type="cylinder" size="0.3 0.01"
          pos="-0.8 0 0" rgba="0.3 0.6 1 0.5" />
    <geom name="position_b" type="cylinder" size="0.3 0.01"
          pos="0.8 0 0" rgba="1 0.6 0.3 0.5" />
</worldbody>
```

### 2. Classroom Environment
Robot as presenter with "audience" objects.

```xml
<worldbody>
    <!-- Whiteboard -->
    <geom name="board" type="box" size="1.5 0.05 1"
          pos="0 1.5 1" rgba="1 1 1 1" />

    <!-- Desk cubes representing students -->
    <body name="desk1" pos="-0.5 -1 0.4">
        <geom type="box" size="0.2 0.3 0.4" rgba="0.6 0.4 0.2 1" />
    </body>
    <body name="desk2" pos="0.5 -1 0.4">
        <geom type="box" size="0.2 0.3 0.4" rgba="0.6 0.4 0.2 1" />
    </body>
</worldbody>
```

### 3. Interaction Test Arena
Objects of different properties for manipulation tests.

```xml
<worldbody>
    <!-- Heavy object -->
    <body name="heavy_box" pos="0.5 0 0.5">
        <geom type="box" size="0.1 0.1 0.1" mass="5.0" rgba="0.5 0.5 0.5 1" />
        <joint type="free" />
    </body>

    <!-- Light object -->
    <body name="light_ball" pos="-0.5 0 0.5">
        <geom type="sphere" size="0.08" mass="0.05" rgba="1 0.2 0.2 1" />
        <joint type="free" />
    </body>

    <!-- Bouncy object -->
    <body name="bouncy_ball" pos="0 0.5 1.5">
        <geom type="sphere" size="0.1" mass="0.1"
              rgba="0.2 1 0.2 1"
              solref="0.01 0.5" />  <!-- High bounciness -->
        <joint type="free" />
    </body>
</worldbody>
```

---

## Camera Control in Viewer

Once the viewer is running:

- **Orbit camera:** Left-click + drag
- **Pan camera:** Right-click + Shift + drag
- **Zoom:** Middle-click + drag OR scroll wheel
- **Reset simulation:** Backspace key

---

## Scene File Locations

**Source scenes (edit these):**
```
/Users/lauras/Desktop/laura/reachy_mini/src/reachy_mini/descriptions/reachy_mini/mjcf/scenes/
```

**Asset files (meshes, textures):**
```
/Users/lauras/Desktop/laura/reachy_mini/src/reachy_mini/descriptions/reachy_mini/mjcf/assets/
```

---

## Testing Workflow

1. **Create scene XML** in `scenes/` folder
2. **Run viewer:** `python3 desktop_viewer.py --scene your_scene_name`
3. **Iterate:**
   - Close viewer
   - Edit XML
   - Re-run viewer
4. **Verify physics:** Press Backspace to reset and watch objects fall

---

## Common Issues

### Scene not found
```
Error: Failed to initialize backend: ...
```
**Solution:** Check scene name matches filename (without `.xml`)

### Mesh not loading
```
Error: Could not open file 'my_model.obj'
```
**Solution:** Verify file is in `assets/` and path in XML is correct

### Objects fall through floor
**Solution:** Make sure floor is at z=0 and objects are above it (positive z)

### Dark/invisible objects
**Solution:** Add a light source:
```xml
<light pos="0 0 3" dir="0 0 -1" directional="true" />
```

---

## Next Steps for Director Agent Integration

1. **Create test scenes** with different configurations
2. **Integrate with scene_manager** - Load MuJoCo scenes based on conversation context
3. **Multi-robot scenes** - Simulate Claude vs Laura debates
4. **Camera recording** - Capture simulation frames for analysis
5. **State synchronization** - Link MuJoCo state to agent memory

---

## Useful Resources

- **MuJoCo Documentation:** https://mujoco.readthedocs.io/
- **MJCF XML Reference:** https://mujoco.readthedocs.io/en/stable/XMLreference.html
- **Example scenes:** Look at `empty.xml` and `minimal.xml` in `scenes/` folder
- **Asset library:** Existing meshes are in `assets/` folder

---

**Tip:** Start simple (just basic shapes), then add complexity. MuJoCo is powerful but syntax-sensitive!
