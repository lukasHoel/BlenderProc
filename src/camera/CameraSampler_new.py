import numbers
import sys
from collections import defaultdict
from pprint import pprint
from typing import Tuple, Dict

import bmesh
import bpy
import bpy_extras as bpye
import mathutils
import numpy as np

from src.camera.CameraInterface import CameraInterface
from src.utility.BlenderUtility import get_all_mesh_objects, get_bounds, world_to_camera, add_cube_based_on_bb
from src.utility.ItemCollection import ItemCollection


class CameraSampler(CameraInterface):
    """ A general camera sampler.

        First a camera pose is sampled according to the configuration, then it is checked if the pose is valid.
        If that's not the case a new camera pose is sampled instead.

        Supported cam pose validation methods:
        - Checking if the distance to objects is in a configured range
        - Checking if the scene coverage/interestingness score is above a configured threshold
        - Checking if a candidate pose is sufficiently different than the sampled poses so far

        Example 1: Sampling 10 camera poses.

        {
          "module": "camera.SuncgCameraSampler",
          "config": {
            "cam_poses": [
            {
              "number_of_samples": 10,
              "proximity_checks": {
                "min": 1.0
              },
              "min_interest_score": 0.4,
              "location": {
                "provider":"sampler.Uniform3d",
                "max":[0, 0, 2],
                "min":[0, 0, 0.5]
              },
              "rotation": {
                "value": {
                  "provider":"sampler.Uniform3d",
                  "max":[1.2217, 0, 6.283185307],
                  "min":[1.2217, 0, 0]
                }
              }
            }
            ]
          }
        }

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "cam_poses", "Camera poses configuration list. Each cell contains a separate config data. Type: list."

    **Properties per cam pose**:

    .. csv-table::
        :header: "Parameter", "Description"

        "number_of_samples", "The number of camera poses that should be sampled. Note depending on some constraints "
                             "(e.g. interest scores), the sampler might not return all of the camera poses if the "
                             "number of tries exceeded the configured limit. Type: int. Default: 1."
        "max_tries", "The maximum number of tries that should be made to sample the requested number of cam poses per "
                     "interest score. Type: int. Default: 100000000."
        "sqrt_number_of_rays", "The square root of the number of rays which will be used to determine, if there is an "
                               "obstacle in front of the camera. Type: int. Default: 10."
        "proximity_checks", "A dictionary containing operators (e.g. avg, min) as keys and as values dictionaries "
                            "containing thresholds in the form of {"min": 1.0, "max":4.0} or just the numerical "
                            "threshold in case of max or min. The operators are combined in conjunction (i.e boolean "
                            "AND). This can also be used to avoid the background in images, with the"
                            "no_background: True option. Type: dict. Default: {}.
        "excluded_objs_in_proximity_check", "A list of objects, returned by getter.Entity to remove some objects from"
                                            "the proximity checks defined in 'proximity_checks'."
                                            "Type: list. Default: []"
        "min_interest_score", "Arbitrary threshold to discard cam poses with less interesting views. Type: float. "
                              "Default: 0.0."
        "interest_score_range", "The maximum of the range of interest scores that would be used to sample the camera "
                                "poses. Interest score range example: min_interest_score = 0.8, interest_score_range = "
                                "1.0, interest_score_step = 0.1 interest score list = [1.0, 0.9, 0.8]. The sampler "
                                "would reject any pose with score less than 1.0. If max tries is reached, it would "
                                "switch to 0.9 and so on. min_interest_score = 0.8, interest_score_range = 0.8, "
                                "interest_score_step = 0.1 (or any value bigger than 0) interest score list = [0.8]. "
                                "Type: float. Default: min_interest_score."
        "interest_score_step", "Step size for the list of interest scores that would be tried in the range from "
                               "min_interest_score to interest_score_range. Must be bigger than 0. Type: float. "
                               "Default: 0.1."
        "special_objects", "Objects that weights differently in calculating whether the scene is interesting or not, "
                           "uses the coarse_grained_class or if not SUNCG, 3D Front, the category_id."
                           "Type: list. Default: []."
        "special_objects_weight", "Weighting factor for more special objects, used to estimate the interestingness of "
                                  "the scene. Type: float. Default: 2.0."
        "check_pose_novelty_rot", "Checks that a sampled new pose is novel with respect to the rotation component. "
                                  "Type: bool. Default: False"
        "check_pose_novelty_translation", "Checks that a sampled new pose is novel with respect to the translation "
                                          "component. Type: bool. Default: False."
        "min_var_diff_rot", "Considers a pose novel if it increases the variance of the rotation component of all poses "
                            "sampled by this parameter's value in percentage. If set to -1, then it would only check "
                            "that the variance is increased. Type: float. Default: sys.float_info.min."
        "min_var_diff_translation", "Same as min_var_diff_rot but for translation. If set to -1, then it would only "
                                    "check that the variance is increased. Type: float. Default: sys.float_info.min."
        "check_if_pose_above_object_list", "A list of objects, where each camera has to be above, could be the floor "
                                           "or a table. Type: list. Default: []."
        "default_cam_param", "A dict which can be used to specify properties across all cam poses. Check CameraInterface "
                             "for more info. Type: dict. Default: {}."
    """

    def __init__(self, config):
        CameraInterface.__init__(self, config)
        self.bvh_tree = None

        self.rotations = []
        self.translations = []

        self.var_rot, self.var_translation   = 0.0, 0.0
        # self.check_pose_novelty_rot = self.config.get_bool("check_pose_novelty_rot", False)
        # self.check_pose_novelty_translation = self.config.get_bool("check_pose_novelty_translation", False)
        self.check_pose_novelty = self.config.get_bool("check_pose_novelty", False)

        self.min_diff_rotation = self.config.get_float("min_diff_rot", sys.float_info.min)
        if self.min_diff_rotation == -1.0:
            self.min_diff_rotation = sys.float_info.min

        self.min_diff_translation = self.config.get_float("min_diff_translation", sys.float_info.min)
        if self.min_diff_translation == -1.0:
            self.min_diff_translation = sys.float_info.min

        self.cam_pose_collection = ItemCollection(self._sample_cam_poses, self.config.get_raw_dict("default_cam_param", {}))

    def run(self):
        """ Sets camera poses. """

        source_specs = self.config.get_list("cam_poses")
        for i, source_spec in enumerate(source_specs):
            self.cam_pose_collection.add_item(source_spec)

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

        all_tries = 0  # max_tries is now applied per each score
        tries = 0

        self.min_interest_score = interest_scores[score_index]
        print("Trying a min_interest_score value: %f" % self.min_interest_score)
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
                    bpy.context.scene.frame_end = frame_id + 1

                    # if frame_id == 0:
                    #     self._visualize_rays(cam, cam_ob.matrix_world)
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

    def sample_and_validate_cam_pose(self, cam, cam_ob, config):
        """ Samples a new camera pose, sets the parameters of the given camera object accordingly and validates it.

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param config: The config object describing how to sample
        :return: True, if the sampled pose was valid
        """
        # Sample/set camera intrinsics
        self._set_cam_intrinsics(cam, config)

        # Sample camera extrinsics (we do not set them yet for performance reasons)
        cam2world_matrix = self._cam2world_matrix_from_cam_extrinsics(config)

        if self._is_pose_valid(cam, cam_ob, cam2world_matrix):
            # Set camera extrinsics as the pose is valid
            cam_ob.matrix_world = cam2world_matrix
            return True
        else:
            return False

    def _is_pose_valid(self, cam, cam_ob, cam2world_matrix):
        """ Determines if the given pose is valid.

        - Checks if the distance to objects is in the configured range
        - Checks if the scene coverage score is above the configured threshold

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param cam2world_matrix: The sampled camera extrinsics in form of a camera to world frame transformation matrix.
        :return: True, if the pose is valid
        """
        if not self._perform_obstacle_in_view_check(cam, cam2world_matrix):
            return False

        if self._is_ceiling_visible(cam, cam2world_matrix):
            # print("Ceiling visible")
            return False

        scene_coverage_score, score, _, coverage_info = self._scene_coverage_score(cam, cam2world_matrix)
        scene_variance, variance_info = self._scene_variance(cam, cam2world_matrix)

        line = [f"Final score: {scene_coverage_score:4.3f}\tScores: ", " | ".join(["{0}: {1}".format(k, v) for k, v in coverage_info.items()])]
        line += [f"Variance: {scene_variance:4.3f}", "\tVariance: ", " | ".join(["{0}: {1}".format(k, v) for k, v in variance_info.items()])]
        if scene_coverage_score < self.min_interest_score or scene_variance < self.min_scene_variance:
            # print(f"\t\t", " ".join(line))
            # pprint("\t", dict(coverage_info), indent=4)
            return False


        objects_are_visible, object_visibilities = self._check_visible_overlap(cam, cam2world_matrix)
        if not objects_are_visible:
            # print("Object overlap too small")
            return False
        line.append("\tVisibility: " + " | ".join([f"{k}-{v[0]:4.3f}/{v[1]:4.3f}" for k, v in object_visibilities.items()]))

        if self.check_pose_novelty and (not self._check_novel_pose(cam2world_matrix)):
            # print("not novel")
            return False

        if self._above_objects:
            for obj in self._above_objects:
                if self._position_is_above_object(cam2world_matrix.to_translation(), obj):
                    return True
            return False

        print(" ".join(line))

        output_path = super()._determine_output_dir() + "/scores.txt"
        with open(output_path, "a") as f:
            f.write(" ".join(line) + "\n")

        return True

    def _position_is_above_object(self, position, object):
        """ Make sure the given position is straight above the given object with no obstacles in between.

        :param position: The position to check.
        :param object: The query object to use.
        :return: True, if a ray sent into negative z-direction starting from the position hits the object first.
        """
        # Send a ray straight down and check if the first hit object is the query object
        hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer,
                                                                 position,
                                                                 mathutils.Vector([0, 0, -1]))
        return hit and hit_object == object

    def _position_is_beneath_object(self, position, object):
        """ Make sure the given position is straight above the given object with no obstacles in between.

        :param position: The position to check.
        :param object: The query object to use.
        :return: True, if a ray sent into negative z-direction starting from the position hits the object first.
        """
        # Send a ray straight down and check if the first hit object is the query object
        hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer,
                                                                 position,
                                                                 mathutils.Vector([0, 0, 1]))
        return hit and hit_object == object


    def _init_bvh_tree(self):
        """ Creates a bvh tree which contains all mesh objects in the scene.

        Such a tree is later used for fast raycasting.
        """
        # Create bmesh which will contain the meshes of all objects
        bm = bmesh.new()
        # Go through all mesh objects
        for obj in get_all_mesh_objects():
            if obj in self.excluded_objects_in_proximity_check or obj.hide_viewport or obj.hide_render:
                continue
            # Add object mesh to bmesh (the newly added vertices will be automatically selected)
            bm.from_mesh(obj.data)
            # Apply world matrix to all selected vertices
            bm.transform(obj.matrix_world, filter={"SELECT"})
            # Deselect all vertices
            for v in bm.verts:
                v.select = False

        # Create tree from bmesh
        self.bvh_tree = mathutils.bvhtree.BVHTree.FromBMesh(bm)

        self._is_bvh_tree_inited = True

    def _perform_obstacle_in_view_check(self, cam, cam2world_matrix):
        """ Check if there is an obstacle in front of the camera which is less than the configured
            "min_dist_to_obstacle" away from it.

        :param cam: The camera whose view frame is used (only FOV is relevant, pose of cam is ignored).
        :param cam2world_matrix: Transformation matrix that transforms from the camera space to the world space.
        :return: True, if there are no obstacles too close to the cam.
        """
        if not self.proximity_checks:  # if no checks are in the settings all positions are accepted
            return True
        if not self._is_bvh_tree_inited:
            raise Exception("The bvh tree should be inited before this function is called!")

        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        sum = 0.0
        sum_sq = 0.0

        range_distance = sys.float_info.max

        # Input validation
        for operator in self.proximity_checks:
            if (operator == "min" or operator == "max") and not isinstance(self.proximity_checks[operator], numbers.Number):
                raise Exception("Threshold must be a number in perform_obstacle_in_view_check")
            if operator == "avg" or operator == "var":
                if "min" not in self.proximity_checks[operator] or "max" not in self.proximity_checks[operator]:
                    raise Exception("Please specify the accepted interval for the avg and var operators "
                                    "in perform_obstacle_in_view_check")
                if not isinstance(self.proximity_checks[operator]["min"], numbers.Number) \
                        or not isinstance(self.proximity_checks[operator]["max"], numbers.Number):
                    raise Exception("Threshold must be a number in perform_obstacle_in_view_check")


        # If there are no average or variance operators, we can decrease the ray range distance for efficiency
        if "avg" not in self.proximity_checks and "var" not in self.proximity_checks:
            if "max" in self.proximity_checks:
                # Cap distance values at a value slightly higher than the max threshold
                range_distance = self.proximity_checks["max"] + 1.0
            else:
                range_distance = self.proximity_checks["min"]

        no_range_distance = False
        if "no_background" in self.proximity_checks and self.proximity_checks["no_background"]:
            # when no background is on, it can not be combined with a reduced range distance
            no_range_distance = True

        # Go in discrete grid-like steps over plane
        position = cam2world_matrix.to_translation()
        for x in range(0, self.sqrt_number_of_rays):
            for y in range(0, self.sqrt_number_of_rays):
                # Compute current point on plane
                end = frame[0] + vec_x * x / float(self.sqrt_number_of_rays - 1) \
                      + vec_y * y / float(self.sqrt_number_of_rays - 1)
                # Send ray from the camera position through the current point on the plane
                if no_range_distance:
                    _, _, _, dist = self.bvh_tree.ray_cast(position, end - position)
                else:
                    _, _, _, dist = self.bvh_tree.ray_cast(position, end - position, range_distance)

                # Check if something was hit and how far it is away
                if dist is not None:
                    if "min" in self.proximity_checks and dist <= self.proximity_checks["min"]:
                        # print(f"sample too close {dist}")
                        return False
                    if "max" in self.proximity_checks and dist >= self.proximity_checks["max"]:
                        # print(f"sample too far {dist}")
                        return False
                    if "avg" in self.proximity_checks:
                        sum += dist
                    if "var" in self.proximity_checks:
                        if not "avg" in self.proximity_checks:
                            sum += dist
                        sum_sq += dist * dist
                elif "no_background" in self.proximity_checks and self.proximity_checks["no_background"]:
                    return False

        if "avg" in self.proximity_checks:
            avg = sum / (self.sqrt_number_of_rays * self.sqrt_number_of_rays)
            # Check that the average distance is not within the accepted interval
            if avg >= self.proximity_checks["avg"]["max"] or avg <= self.proximity_checks["avg"]["min"]:
                return False

        if "var" in self.proximity_checks:
            if not "avg" in self.proximity_checks:
                avg = sum / (self.sqrt_number_of_rays * self.sqrt_number_of_rays)
            sq_avg = avg * avg

            avg_sq = sum_sq / (self.sqrt_number_of_rays * self.sqrt_number_of_rays)

            var = avg_sq - sq_avg
            # Check that the variance value of the distance is not within the accepted interval
            if var >= self.proximity_checks["var"]["max"] or var <= self.proximity_checks["var"]["min"]:
                return False

        return True

    def _is_ceiling_visible(self, cam, cam2world_matrix):
        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        position = cam2world_matrix.to_translation()

        for x in range(0, self.sqrt_number_of_rays):
            for y in range(0, self.sqrt_number_of_rays):
                x_ratio = x / float(self.sqrt_number_of_rays - 1)
                y_ratio = y / float(self.sqrt_number_of_rays - 1)
                end = frame[0] + vec_x * x_ratio + vec_y * y_ratio
                # start = end - offset
                start = position

                # Send ray from the camera position through the current point on the plane
                hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, start, end-start)

                if hit and "nyu_category_id" in hit_object and hit_object["nyu_category_id"] == 22:
                    return True

        return False

    def _visualize_rays(self, cam, cam2world_matrix, center_only=False):
        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        x_offset = vec_x * self.center_region_x_percentage / 2
        vec_x_center = vec_x * self.center_region_x_percentage

        y_offset = vec_y * self.center_region_y_percentage / 2
        vec_y_center = vec_y * self.center_region_y_percentage

        # Go in discrete grid-like steps over plane
        position = cam2world_matrix.to_translation()
        # bpy.ops.mesh.primitive_ico_sphere_add(location=position, radius=0.05)

        center_point = frame[0] + (frame[2] - frame[0]) / 2
        offset = center_point - position

        for x in range(0, self.sqrt_number_of_rays):
            for y in range(0, self.sqrt_number_of_rays):
                x_ratio = x / float(self.sqrt_number_of_rays - 1)
                y_ratio = y / float(self.sqrt_number_of_rays - 1)

                if center_only:
                    end = frame[0] + x_offset + y_offset + vec_x_center * x_ratio + vec_y_center * y_ratio
                    start = position #end - offset
                    ray_start = start #end
                else:
                    end = frame[0] + vec_x * x_ratio + vec_y * y_ratio
                    start = position
                    ray_start = start
                # Send ray from the camera position through the current point on the plane
                hit, hit_location, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, ray_start, end-start)
                # bpy.ops.mesh.primitive_cube_add(location=start, size=0.02)

                if hit:
                    bpy.ops.mesh.primitive_cube_add(location=hit_location, size=0.03)
                    if "coarse_grained_class" in hit_object:
                        object_class = hit_object["coarse_grained_class"]
                        # print(object_class)

                bpy.ops.mesh.primitive_ico_sphere_add(location=end, radius=0.02)


    def _scene_variance(self, cam, cam2world_matrix):
        num_of_rays = self.sqrt_number_of_rays * self.sqrt_number_of_rays
        objects_hit = defaultdict(int)

        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        position = cam2world_matrix.to_translation()
        center_point = frame[0] + (frame[2] - frame[0]) / 2
        offset = center_point - position

        # Go in discrete grid-like steps over plane
        for x in range(0, self.sqrt_number_of_rays):
            for y in range(0, self.sqrt_number_of_rays):
                # Compute current point on plane
                x_ratio = x / float(self.sqrt_number_of_rays - 1)
                y_ratio = y / float(self.sqrt_number_of_rays - 1)
                end = frame[0] + vec_x * x_ratio + vec_y * y_ratio
                # start = end - offset
                # Send ray from the camera position through the current point on the plane
                hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, position, end - position)

                if hit:
                    is_of_special_dataset = "is_suncg" in hit_object or "is_3d_front" in hit_object
                    if is_of_special_dataset and "type" in hit_object and hit_object["type"] == "Object":
                        # calculate the score based on the type of the object,
                        # wall, floor and ceiling objects have 0 score
                        if "nyu_category_id" in hit_object and hit_object["nyu_category_id"] not in self.excluded_objects_in_score_check:
                            object_class = hit_object["nyu_category_id"]
                            objects_hit[object_class] += 1

        # For a scene with three different objects, the starting variance is 1.0, increases/decreases by '1/3' for
        # each object more/less, excluding floor, ceiling and walls
        scene_variance = len(objects_hit) / 3.0
        for object_hit_value in objects_hit.values():
            # For an object taking half of the scene, the scene_variance is halved, this penalizes non-even
            # distribution of the objects in the scene
            scene_variance *= 1.0 - object_hit_value / float(num_of_rays)
        return scene_variance, objects_hit

    def _scene_coverage_score(self, cam, cam2world_matrix):
        """ Evaluate the interestingness/coverage of the scene.

        This module tries to look at as many objects at possible, this might lead to
        a focus on the same objects from similar angles.

        Only for SUNCG and 3D Front:
            Least interesting objects: walls, ceilings, floors.

        :param cam: The camera whose view frame is used (only FOV is relevant, pose of cam is ignored).
        :param cam2world_matrix: The world matrix which describes the camera orientation to check.
        :return: the scoring of the scene.
        """

        num_of_rays = self.sqrt_number_of_rays * self.sqrt_number_of_rays
        score = 0.0
        objects_hit = defaultdict(int)
        objects_score = defaultdict(int)

        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        x_offset = vec_x * self.center_region_x_percentage / 2
        vec_x_center = vec_x * self.center_region_x_percentage

        y_offset = vec_y * self.center_region_y_percentage / 2
        vec_y_center = vec_y * self.center_region_y_percentage

        position = cam2world_matrix.to_translation()
        center_point = frame[0] + (frame[2] - frame[0]) / 2
        offset = center_point - position

        # Go in discrete grid-like steps over plane
        for x in range(0, self.sqrt_number_of_rays):
            for y in range(0, self.sqrt_number_of_rays):
                # Compute current point on plane
                x_ratio = x / float(self.sqrt_number_of_rays - 1)
                y_ratio = y / float(self.sqrt_number_of_rays - 1)
                end = frame[0] + x_offset + y_offset + vec_x_center * x_ratio + vec_y_center * y_ratio
                start = end - offset
                # Send ray from the camera position through the current point on the plane
                # hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, end, end-start)
                hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, position, end-position)

                if hit:
                    is_of_special_dataset = "is_suncg" in hit_object or "is_3d_front" in hit_object
                    if is_of_special_dataset and "type" in hit_object and hit_object["type"] == "Object":
                        # calculate the score based on the type of the object,
                        # wall, floor and ceiling objects have 0 score
                        if "nyu_category_id" in hit_object:
                            object_class = hit_object["nyu_category_id"]

                            objects_hit[object_class] += 1
                            if hit_object.get("nyu_category_id", -1) in self.excluded_objects_in_score_check:
                                score += 0
                            else:
                                objects_score[object_class] += 1
                                if object_class in self.special_objects:
                                    score += self.special_objects_weight
                                else:
                                    score += 1
                        else:
                            score += 1

                        # if self._check_center_region(x, y, center_region):
                        #     score += self.center_region_weight

                    # elif "category_id" in hit_object:
                    #     object_class = hit_object["category_id"]
                    #     if object_class in self.special_objects:
                    #         score += self.special_objects_weight
                    #     else:
                    #         score += 1
                    #     objects_hit[object_class] += 1
                    # else:
                    #     objects_hit[hit_object] += 1
                    #     score += 1
                else:
                    return 0, 0, 0, {}

        # For a scene with three different objects, the starting variance is 1.0, increases/decreases by '1/3' for
        # each object more/less, excluding floor, ceiling and walls
        scene_variance = len(objects_hit) / 3.0
        for object_hit_value in objects_hit.values():
            # For an object taking half of the scene, the scene_variance is halved, this penalizes non-even
            # distribution of the objects in the scene
            scene_variance *= 1.0 - object_hit_value / float(num_of_rays)
        score_normalized = score / float(num_of_rays)
        score_factored = scene_variance * score_normalized
        return score_normalized, score_normalized, scene_variance, objects_score

    def _check_center_region(self, x, y, center_region):
        is_horizontal = center_region[0] < x < center_region[1]
        is_vertical = center_region[2] < y < center_region[3]

        return is_horizontal and is_vertical

    def _check_visible_overlap(self, cam, cam2world_matrix) -> Tuple[bool, Dict]:
        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        position = cam2world_matrix.to_translation()

        objects_hit = {}

        num_rays = self.sqrt_number_of_rays

        for x in range(0, num_rays):
            for y in range(0, num_rays):
                # Compute current point on plane
                end = frame[0] + vec_x * x / float(num_rays - 1) \
                      + vec_y * y / float(num_rays - 1)
                # Send ray from the camera position through the current point on the plane
                hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, position, end - position)

                if hit and "nyu_category_id" in hit_object and hit_object["nyu_category_id"] not in self.excluded_objects_in_overlap_check:
                    model_id = hit_object.name
                    if model_id not in objects_hit:
                        objects_hit[model_id] = hit_object

        world2camera_matrix = cam2world_matrix.inverted()

        view_frame = cam.view_frame(scene=bpy.context.scene)
        view_frame_center = (view_frame[2] - view_frame[0]) / 2

        position_view = world_to_camera(cam, position, world2camera_matrix)

        object_visibilities = {}
        index = 0
        for model_id, obj in objects_hit.items():
            bounding_box = get_bounds(obj)
            bounding_box_view = np.array([world_to_camera(cam, v, world2camera_matrix) for v in bounding_box])

            min_coord = np.min(bounding_box_view, axis=0)
            max_coord = np.max(bounding_box_view, axis=0)
            # min_coord[2] = -position_view[2]
            # max_coord[2] = -position_view[2]

            min_coord_world = cam2world_matrix @ mathutils.Vector(min_coord)
            max_coord_world = cam2world_matrix @ mathutils.Vector(max_coord)

            min_coord_visible = np.clip(min_coord, a_min=0, a_max=None)
            max_coord_visible = np.clip(max_coord, a_max=1, a_min=None)

            min_coord_visible_world = cam2world_matrix @ mathutils.Vector(min_coord_visible)
            max_coord_visible_world = cam2world_matrix @ mathutils.Vector(max_coord_visible)

            extent_bbox = abs(max_coord - min_coord)
            extent_visible = abs(max_coord_visible - min_coord_visible)

            extent_ratio = extent_visible / extent_bbox
            # object_visibilities[model_id] = extent_ratio

            if extent_ratio[0] < self.min_visible_overlap or extent_ratio[1] < self.min_visible_overlap:
                print(f"{model_id}: {extent_ratio} - Skip frame because overlap is too small")
                return False, object_visibilities
            else:
                object_visibilities[model_id] = extent_ratio
            #
            # if index == 0:
            #     bpy.ops.mesh.primitive_ico_sphere_add(location=bbox_min, radius=0.05)
            #     bpy.ops.mesh.primitive_ico_sphere_add(location=bbox_max, radius=0.05)
            #
            #     bpy.ops.mesh.primitive_cube_add(location=min_coord_world, size=0.05)
            #     bpy.ops.mesh.primitive_cube_add(location=max_coord_world, size=0.05)
            #
            #     bpy.ops.mesh.primitive_cylinder_add(location=min_coord_world_visible, radius=0.05)
            #     bpy.ops.mesh.primitive_cylinder_add(location=max_coord_world_visible, radius=0.05)
            #     # bpy.ops.mesh.primitive_cone_add(location=max_coord_visible_world, radius1=0.05)
            # index += 1

        # self._visualize_rays(cam, cam2world_matrix)

        return True, object_visibilities



    def _check_novel_pose(self, cam2world_matrix):
        """ Checks if a newly sampled pose is novel based on variance checks.

        :param cam2world_matrix: camera pose to check
        """

        # def _variance_constraint(array, new_val, old_var, diff_threshold, mode):
        #     array.append(new_val)
        #     var = np.var(array, axis=0)
        #
        #     if np.any(var < 0.5 * old_var):
        #         array.pop()
        #         return False
        #
        #     diff = ((var - old_var) / old_var) * 100.0
        #     print("Variance difference {}: {}".format(mode, diff))
        #     if any(diff < diff_threshold):  # Check if the variance increased sufficiently
        #         array.pop()
        #         return False
        #
        #     return True

        translation = cam2world_matrix.to_translation()
        rotation    = cam2world_matrix.col[2].normalized()

        if len(self.translations) > 0 and len(self.rotations) > 0:  # First pose is always novel

            if self.check_pose_novelty:
                for previous_translation, previous_rotation in zip(self.translations, self.rotations):
                    diff_translation = (previous_translation - translation).length

                    if diff_translation < self.min_diff_translation:
                        # print(f"Translation difference too small {diff_translation}")

                        diff_dot = 1 - previous_rotation.dot(rotation)

                        if diff_dot < self.min_diff_rotation:
                            # print(f"Rotation difference too small {diff_dot}")
                            return False

            # if self.check_pose_novelty_rot:
            #     if not _variance_constraint(self.rotations, rotation, self.var_rot, self.min_var_diff_rot, "rotation"):
            #         return False
            #
            # if self.check_pose_novelty_translation:
            #     if not _variance_constraint(self.translations, translation, self.var_translation,
            #                                 self.min_var_diff_translation, "translation"):
            #         return False

        self.translations.append(translation)
        self.rotations.append(rotation)

        # self.var_rot = np.var(self.rotations, axis=0)
        # self.var_translation = np.var(self.translations, axis=0)

        return True 
