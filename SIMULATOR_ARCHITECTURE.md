# MuJoCo Sitcom Scene Simulator: Project Architecture

## 1. Project Goal

This project's goal is to build a flexible, modular simulation environment using MuJoCo. The system should allow a user or AI agent to easily generate and run various scenes within a virtual "sitcom apartment" set, populated with a library of characters and props.

## 2. My Role as the AI Agent

My primary role is to act as the **Scene Operator**. I do not create the base 3D assets, but I use the project's architecture to generate and launch new scenes based on high-level user requests.

My core tasks are:
1.  **Discover Assets:** List available props and characters from the Asset Library.
2.  **Write Scene Scripts:** Generate new `.yaml` files that define the placement of assets for a scene.
3.  **Execute the Simulation:** Run the master Python script to build the MuJoCo XML and launch the viewer.

## 3. Core Architecture

The project uses a 4-part system that separates assets from layout and scene definition.

### a. The Asset Library
*   **Location:** A root `/assets/` directory.
*   **Purpose:** Contains all reusable 3D models and their corresponding MuJoCo XML component files.
*   **Structure:**
    *   `/assets/meshes/`: Raw `.stl` or `.obj` 3D model files.
    *   `/assets/mjcf/props/`: XML components for props (e.g., `sofa.xml`, `mug.xml`).
    *   `/assets/mjcf/characters/`: XML components for agents (e.g., `reachy.xml`).
*   **Rule:** All XML files in the asset library are designed as modular components (they do not contain a `<worldbody>`).

### b. The Floorplan
*   **Format:** A `.yaml` or `.json` file that abstractly defines the static walls and layout of the set.
*   **Example (`apartment_floorplan.yaml`):**
    ```yaml
    name: "The Apartment"
    walls:
      - { type: "wall_segment", start: [0,0], end: [5,0] }
      - { type: "wall_with_door", start: [5,0], end: [5,5], door_at: 2.5 }
    ```

### c. The Scene Script
*   **Format:** A `.yaml` file describing the specific placement of assets for one scene. **This is the main file I will create.**
*   **Example (`morning_coffee_scene.yaml`):**
    ```yaml
    floorplan: "apartment_floorplan"
    placements:
      - { asset: "sofa", position: [2.5, 2, 0], rotation: 180 }
      - { asset: "reachy", position: [2.5, 2.5, 0.5], rotation: -90 }
    ```

### d. The Scene Generator ("The Director")
*   **File:** `scene_manager.py`
*   **Function:** A master Python script that takes a Scene Script as input, reads the appropriate files from the Asset Library and Floorplan, and programmatically generates the final, complete `.xml` file that MuJoCo can run.

## 4. Key Commands & Workflow

This is the process I will follow:

1.  **User Request:** "Create a scene with Reachy on the sofa."
2.  **My Action (Discover):** Check for available assets.
    *   `ls -R assets/mjcf`
3.  **My Action (Write):** Generate a new scene script.
    *   `write_file('scenes/new_scene.yaml', ...)`
4.  **My Action (Execute):** Run the Scene Generator to build and launch the simulation.
    *   `run_shell_command('python3 scene_manager.py --scene scenes/new_scene.yaml')`
