# (Optional) How to customize the environment

If you’d like to customize the game level, open the level scene `res://scenes/level.tscn`, then open the `res://scenes/modules/` folder in the Godot FileSystem:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit13/level_scene.jpg" alt="level scene"/>

The level contains 3 rooms made using the modules, robot, and some additional colliders which prevent the ability to complete the level by climbing on a wall in the first room and reaching the key that way. By adding the modules to the scene, you can add new rooms and items.

If you click on the Key node (it’s in `Room3`, you can also search for it), then click on `Node > Signals`, you will see that the `collected` signal is connected to both the robot and the chest. We use this to track whether the robot has collected the key, and to unlock the chest. The same system is applied for using the lever to activate the stairs, and if you add more levers/stairs/keys, you can connect them using signals.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit13/level_signals.jpg" alt="level signals"/>

If you switch to `Groups`, you will see that the key is a member of the `resetable` group. In the same group we have the raft, lever, chest, player, and can add any node that needs to be reset when the episode resets.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit13/resetables_group.jpg" alt="resetables group"/>

For this to work, every object that is in the `resetable` group also needs to implement the `reset()` method, which takes care of resetting that object.

Because we have multiple instances of the level scene for training, we don’t reset all `resetables`, but only those within the same scene. In `level_manager.gd`, we have a method `reset_all_resetables()` that takes care of this, and it is called by the robot script when resetting is needed.

After changing the level size, updating the `level_size` variable in `robot_ai_controller.gd` is also needed. For this, just roughly measure the longest dimension of the level, and update the variable.

If you change the amount of objects that need to be tracked by the `AIController` (levers, rafts, etc.), you will need to update the relevant code in the script, include export properties for those objects, then connect them in the inspector properties of `AIController` in the level scene:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit13/ai_controller_inspector_properties.jpg" alt="ai controller inspector properties"/>

After this, you may also need to update the same properties of the `AIController` in the demo record scene as well.