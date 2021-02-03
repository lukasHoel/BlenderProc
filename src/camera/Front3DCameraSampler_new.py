import json
import random
from pathlib import Path

import numpy as np

import bpy

from src.camera.CameraSampler import CameraSampler
from src.utility.BlenderUtility import get_bounds, hide_all_geometry, show_collection


class Front3DCameraSampler(CameraSampler):
    """
    This Camera Sampler is similar to how the SuncgCameraSampler works.

    It first searches for rooms, by using the different floors, which are used in each room.
    It then counts the amount of 3D-Future objects on this particular floor, to check if this room is interesting
    for creating cameras or not. The amount of needed objects can be changed via the config.
    If the amount is set to 0, all rooms will have cameras, even if these rooms are empty.

    The Front3D Loader provides information for using the min_interesting_score option.
    Furthermore, it supports the no_background: True option, which is useful as the 3D-Front dataset has no windows
    or doors to the outside world, which then leads to the background appearing in this shots, if not activated.

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"
        "amount_of_objects_needed_per_room", "The amount of objects needed per room, so that cameras are sampled in it.
                                             "This avoids that cameras are sampled in empty rooms."
                                             "Type: int. Default: 2"
    """

    def __init__(self, config):
        CameraSampler.__init__(self, config)
        self.used_floors = []


    def run(self):
        # all_objects = get_all_mesh_objects()
        # front_3D_objs = [obj for obj in all_objects if "is_3D_future" in obj and obj["is_3D_future"]]
        #
        # floor_objs = [obj for obj in front_3D_objs if obj.name.lower().startswith("floor")]
        #
        # # count objects per floor -> room
        # floor_obj_counters = {obj.name: 0 for obj in floor_objs}
        # counter = 0
        # for obj in front_3D_objs:
        #     name = obj.name.lower()
        #     if "wall" in name or "ceiling" in name:
        #         continue
        #     counter += 1
        #     location = obj.location
        #     for floor_obj in floor_objs:
        #         is_above = self._position_is_above_object(location, floor_obj)
        #         if is_above:
        #             floor_obj_counters[floor_obj.name] += 1
        # amount_of_objects_needed_per_room = self.config.get_int("amount_of_objects_needed_per_room", 2)
        # self.used_floors = [obj for obj in floor_objs if floor_obj_counters[obj.name] > amount_of_objects_needed_per_room]
        amount_of_objects_needed_per_room = self.config.get_int("amount_of_objects_needed_per_room", 1)
        self.rooms = {}
        for room_obj in bpy.context.scene.objects:
            # Check if object is from type room and has bbox
            if "is_room" in room_obj and room_obj["is_room"] == 1:
                # count objects
                room_objects = [obj for obj in room_obj.children if "is_3D_future" in obj and obj["is_3D_future"] == 1]
                num_room_objects = len(room_objects)

                floors = list(filter(lambda x: x.name.lower().startswith("floor"), room_obj.children))

                if len(floors) == 0:
                    print(f"Skip {room_obj.name}: 0 floor objects found")
                    continue

                if len(floors) > 1 or len(floors) == 0:
                    print(f"Skip {room_obj.name}: {len(floors)} floor objects found")
                    continue

                floor = floors[0]

                if "num_floors" in floor and floor["num_floors"] > 2:
                    print(f"Skip {room_obj.name}: Too many floors merged ({floor['num_floors']})")
                    continue

                if num_room_objects < amount_of_objects_needed_per_room:
                    print(f"Skip {room_obj.name}: Not enough objects in room ({num_room_objects})")
                    continue

                self.rooms[room_obj.name] = room_obj, floor, room_obj["room_id"]

        output_path = Path(super()._determine_output_dir()) / f"room_mapping.json"
        with open(output_path, "w") as f:
            name_index_mapping = {obj[2]: name for name, obj in self.rooms.items()}
            json.dump(name_index_mapping, f, indent=4)

        print(f"Found {len(self.rooms)} rooms")
        super().run()

    def sample_and_validate_cam_pose(self, cam, cam_ob, config):
        """ Samples a new camera pose, sets the parameters of the given camera object accordingly and validates it.

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param config: The config object describing how to sample
        :return: True, if the sampled pose was valid
        """
        # Sample room
        # floor_obj = random.choice(self.used_floors)
        room_obj, floor_obj, room_index = self.rooms[self.current_room_name]

        # Sample/set intrinsics
        self._set_cam_intrinsics(cam, config)

        # Sample camera extrinsics (we do not set them yet for performance reasons)
        cam2world_matrix = self._cam2world_matrix_from_cam_extrinsics(config)

        # Make sure the sampled location is inside the room => overwrite x and y and add offset to z
        bounding_box = get_bounds(floor_obj)
        min_corner = np.min(bounding_box, axis=0)
        max_corner = np.max(bounding_box, axis=0)

        cam2world_matrix.translation[0] = random.uniform(min_corner[0], max_corner[0])
        cam2world_matrix.translation[1] = random.uniform(min_corner[1], max_corner[1])
        cam2world_matrix.translation[2] += floor_obj.location[2]

        # Check if sampled pose is valid
        if self._is_pose_valid(floor_obj, cam, cam_ob, cam2world_matrix):
            # Set camera extrinsics as the pose is valid
            cam_ob.matrix_world = cam2world_matrix

            cam_ob["room_id"] = room_index
            return True
        else:
            return False

    def _sample_cam_poses(self, config):
        """ Samples camera poses according to the given config

        :param config: The config object
        """
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data

        # Set global parameters
        self._is_bvh_tree_inited = False
        self.sqrt_number_of_rays = config.get_int("sqrt_number_of_rays", 10)
        self.max_tries = config.get_int("max_tries", 100000000)
        self.proximity_checks = config.get_raw_dict("proximity_checks", {})
        self.excluded_objects_in_proximity_check = config.get_list("excluded_objs_in_proximity_check", [])
        self.min_visible_overlap = config.get_float("min_visible_overlap", 0.0)
        self.min_scene_variance = config.get_float("min_scene_variance", 0.0)
        self.min_interest_score = config.get_float("min_interest_score", 0.0)
        self.interest_score_range = config.get_float("interest_score_range", self.min_interest_score)
        self.interest_score_step = config.get_float("interest_score_step", 0.1)
        self.special_objects = config.get_list("special_objects", [])
        self.special_objects_weight = config.get_float("special_objects_weight", 2)
        self.excluded_objects_in_score_check = config.get_list("excluded_objects_in_score_check", [])
        self.excluded_objects_in_overlap_check = config.get_list("excluded_objects_in_overlap_check", [])
        self.center_region_x_percentage = config.get_float("center_region_x_percentage", 0.5)
        self.center_region_y_percentage = config.get_float("center_region_y_percentage", 0.5)
        self.center_region_weight = config.get_float("center_region_weight", 5)
        self._above_objects = config.get_list("check_if_pose_above_object_list", [])

        if self.proximity_checks:
            # needs to build an bvh tree
            self._init_bvh_tree()

        if self.interest_score_step <= 0.0:
            raise Exception("Must have an interest score step size bigger than 0")

        # Determine the number of camera poses to sample
        number_of_poses = config.get_int("number_of_samples", 1)  # num samples per room
        print("Sampling " + str(number_of_poses) + " cam poses")

        if self.min_interest_score == self.interest_score_range:
            step_size = 1
        else:
            step_size = (self.interest_score_range - self.min_interest_score) / self.interest_score_step
            step_size += 1  # To include last value
        # Decreasing order
        interest_scores = np.linspace(self.interest_score_range, self.min_interest_score, step_size)
        score_index = 0

        self.min_interest_score = interest_scores[score_index]
        print("Trying a min_interest_score value: %f" % self.min_interest_score)
        for room_name, (room_obj, _, _) in self.rooms.items():
            print(f"Sample views in {room_obj.name}")
            all_tries = 0  # max_tries is now applied per each score
            tries = 0
            self.current_room_name = room_name

            # hide everything except current room
            print("Hide geometry")
            hide_all_geometry()
            print("display single room")
            show_collection(room_obj)
            bpy.context.view_layer.update()

            print("start sampling")
            for i in range(number_of_poses):
                # Do until a valid pose has been found or the max number of tries has been reached
                fraction_tries = self.max_tries // 10

                while tries < self.max_tries:
                    if tries % fraction_tries == 0:
                        print(f"Performed {tries} tires")
                    tries += 1
                    all_tries += 1
                    # Sample a new cam pose and check if its valid
                    if self.sample_and_validate_cam_pose(cam, cam_ob, config):
                        # Store new cam pose as next frame
                        frame_id = bpy.context.scene.frame_end
                        self._insert_key_frames(cam, cam_ob, frame_id)
                        self.insert_geometry_key_frame(room_obj, frame_id)
                        bpy.context.scene.frame_end = frame_id + 1

                        # if frame_id == 0:
                        # self._visualize_rays(cam, cam_ob.matrix_world, center_only=True)
                        break

                if tries >= self.max_tries:
                    if score_index == len(interest_scores) - 1:  # If we tried all score values
                        print(f"Maximum number of tries reached! Found: {bpy.context.scene.frame_end} poses")
                        break
                    # Otherwise, try a different lower score and reset the number of trials
                    score_index += 1
                    self.min_interest_score = interest_scores[score_index]
                    print("Trying a different min_interest_score value: %f" % self.min_interest_score)
                    tries = 0

            print(str(all_tries) + " tries were necessary")

    def _insert_key_frames(self, cam, cam_ob, frame_id):
        """ Insert key frames for all relevant camera attributes.

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param frame_id: The frame number where key frames should be inserted.
        """
        # As the room id depends on the camera pose and therefore on the keyframe, we also need to add keyframes for the room id
        cam_ob.keyframe_insert(data_path='["room_id"]', frame=frame_id)

        # Set visibility key frames for all objects
        # Select all objects except current room, set obj.hide_render = True
        # obj.keyframe_insert(data_path='hide_render', frame=frame_id)

        # Add the usual key frames
        super()._insert_key_frames(cam, cam_ob, frame_id)

    def _is_pose_valid(self, floor_obj, cam, cam_ob, cam2world_matrix):
        """ Determines if the given pose is valid.

        - Checks if the pose is above the floor
        - Checks if the distance to objects is in the configured range
        - Checks if the scene coverage score is above the configured threshold

        :param floor_obj: The floor object of the room the camera was sampled in.
        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param cam2world_matrix: The sampled camera extrinsics in form of a camera to world frame transformation matrix.
        :return: True, if the pose is valid
        """
        if not self._position_is_above_object(cam2world_matrix.to_translation(), floor_obj):
            return False

        return super()._is_pose_valid(cam, cam_ob, cam2world_matrix)

    def insert_geometry_key_frame(self, room_obj, frame_id):
        for room_name, (room, _, _) in self.rooms.items():
            should_hide = room_obj.name != room_name
            for obj in room.children:
                obj.hide_viewport = should_hide
                obj.hide_render = should_hide
                obj.keyframe_insert(data_path='hide_viewport', frame=frame_id)
                obj.keyframe_insert(data_path='hide_render', frame=frame_id)

