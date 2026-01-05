#!/usr/bin/env python3
"""
Command Pattern Implementation for Undo/Redo System
"""

class Command:
    """Base class for all commands that can be undone/redone"""

    def execute(self):
        """Execute the command"""
        pass

    def undo(self):
        """Undo the command"""
        pass

    def redo(self):
        """Redo the command (usually same as execute)"""
        self.execute()

    def description(self):
        """Return a human-readable description of the command"""
        return "Command"


class CommandStack:
    """Manages the undo/redo stack"""

    def __init__(self, max_size=100):
        self.stack = []
        self.index = -1  # Points to the last executed command
        self.max_size = max_size

    def execute(self, command):
        """Execute a command and add it to the stack"""
        # Remove any commands after the current index (when doing new action after undo)
        self.stack = self.stack[:self.index + 1]

        # Execute the command
        command.execute()

        # Add to stack
        self.stack.append(command)
        self.index += 1

        # Maintain max size
        if len(self.stack) > self.max_size:
            self.stack.pop(0)
            self.index -= 1

    def undo(self):
        """Undo the last command"""
        if self.can_undo():
            command = self.stack[self.index]
            command.undo()
            self.index -= 1
            return command.description()
        return None

    def redo(self):
        """Redo the next command"""
        if self.can_redo():
            self.index += 1
            command = self.stack[self.index]
            command.redo()
            return command.description()
        return None

    def can_undo(self):
        """Check if undo is available"""
        return self.index >= 0

    def can_redo(self):
        """Check if redo is available"""
        return self.index < len(self.stack) - 1

    def clear(self):
        """Clear the command stack"""
        self.stack = []
        self.index = -1


# Specific Command Implementations

class AddControlPointCommand(Command):
    """Command for adding a control point"""

    def __init__(self, model, position):
        self.model = model
        self.position = position
        self.point_index = None

    def execute(self):
        self.point_index = len(self.model.control_points)
        self.model.control_points.append(list(self.position))
        self.model.selected_points.add(self.point_index)

    def undo(self):
        if self.point_index is not None:
            self.model.control_points.pop(self.point_index)
            self.model.selected_points.discard(self.point_index)
            # Update indices of remaining points
            for i in range(self.point_index, len(self.model.control_points)):
                if i + 1 in self.model.selected_points:
                    self.model.selected_points.remove(i + 1)
                    self.model.selected_points.add(i)

    def description(self):
        return f"Add control point at {self.position}"


class DeleteControlPointCommand(Command):
    """Command for deleting selected control points"""

    def __init__(self, model):
        self.model = model
        self.deleted_points = []
        self.deleted_indices = []

    def execute(self):
        # Sort indices in descending order to delete from end
        indices_to_delete = sorted(self.model.selected_points, reverse=True)
        for index in indices_to_delete:
            if index < len(self.model.control_points):
                self.deleted_points.append((index, self.model.control_points[index][:]))
                self.deleted_indices.append(index)
                self.model.control_points.pop(index)

        self.model.selected_points.clear()

    def undo(self):
        # Restore points in reverse order
        for index, point in reversed(self.deleted_points):
            self.model.control_points.insert(index, point[:])

    def description(self):
        return f"Delete {len(self.deleted_indices)} control point(s)"


class MoveControlPointCommand(Command):
    """Command for moving control points"""

    def __init__(self, model, delta):
        self.model = model
        self.delta = delta
        self.old_positions = {}

    def execute(self):
        self.old_positions.clear()
        for index in self.model.selected_points:
            if index < len(self.model.control_points):
                self.old_positions[index] = self.model.control_points[index][:]
                self.model.control_points[index][0] += self.delta[0]
                self.model.control_points[index][1] += self.delta[1]
                self.model.control_points[index][2] += self.delta[2]

    def undo(self):
        for index, old_pos in self.old_positions.items():
            self.model.control_points[index] = old_pos[:]

    def description(self):
        return f"Move {len(self.old_positions)} control point(s)"


class CreatePatchCommand(Command):
    """Command for creating a patch from selected points"""

    def __init__(self, model):
        self.model = model
        self.patch_index = None

    def execute(self):
        if len(self.model.selected_points) == 16:
            points = [self.model.control_points[i] for i in sorted(self.model.selected_points)]
            self.patch_index = len(self.model.patches)
            self.model.patches.append(points)
            self.model.selected_points.clear()

    def undo(self):
        if self.patch_index is not None:
            self.model.patches.pop(self.patch_index)

    def description(self):
        return "Create patch from 16 control points"


class AddBoneCommand(Command):
    """Command for adding a bone"""

    def __init__(self, model, name, position, parent_name=None):
        self.model = model
        self.name = name
        self.position = position
        self.parent_name = parent_name
        self.bone = None

    def execute(self):
        from main import Bone  # Import here to avoid circular import
        parent = None
        if self.parent_name:
            parent = self.model.bones.get(self.parent_name)

        self.bone = Bone(self.name, self.position, parent=parent)
        if parent:
            parent.children.append(self.bone)

        self.model.bones[self.name] = self.bone

    def undo(self):
        if self.bone:
            del self.model.bones[self.name]
            if self.bone.parent:
                self.bone.parent.children.remove(self.bone)

    def description(self):
        return f"Add bone '{self.name}'"


class MoveBoneCommand(Command):
    """Command for moving a bone"""

    def __init__(self, model, bone_name, new_position, new_rotation):
        self.model = model
        self.bone_name = bone_name
        self.new_position = new_position
        self.new_rotation = new_rotation
        self.old_position = None
        self.old_rotation = None

    def execute(self):
        bone = self.model.find_bone_by_name(self.bone_name)
        if bone:
            self.old_position = bone.position[:]
            self.old_rotation = bone.rotation[:]
            bone.position = list(self.new_position)
            bone.rotation = list(self.new_rotation)

    def undo(self):
        bone = self.model.find_bone_by_name(self.bone_name)
        if bone and self.old_position and self.old_rotation:
            bone.position = self.old_position[:]
            bone.rotation = self.old_rotation[:]

    def description(self):
        return f"Move bone '{self.bone_name}'"


class CreateMaterialCommand(Command):
    """Command for creating a new material"""

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def execute(self):
        if self.name not in self.model.materials:
            self.model.create_material(self.name)

    def undo(self):
        if self.name in self.model.materials:
            del self.model.materials[self.name]

    def description(self):
        return f"Create material '{self.name}'"


class AssignMaterialCommand(Command):
    """Command for assigning material to an object"""

    def __init__(self, model, object_id, material_name):
        self.model = model
        self.object_id = object_id
        self.new_material = material_name
        self.old_material = model.material_assignments.get(object_id)

    def execute(self):
        self.model.assign_material(self.object_id, self.new_material)

    def undo(self):
        if self.old_material:
            self.model.assign_material(self.object_id, self.old_material)
        elif self.object_id in self.model.material_assignments:
            del self.model.material_assignments[self.object_id]

    def description(self):
        return f"Assign material '{self.new_material}' to object '{self.object_id}'"


class CreateAnimationCommand(Command):
    """Command for creating a new animation"""

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def execute(self):
        if self.name not in self.model.animations:
            from main import Animation  # Import here to avoid circular import
            self.model.animations[self.name] = Animation(self.name)

    def undo(self):
        if self.name in self.model.animations:
            del self.model.animations[self.name]

    def description(self):
        return f"Create animation '{self.name}'"


class AddKeyframeCommand(Command):
    """Command for adding a keyframe to an animation"""

    def __init__(self, model, animation_name, time, bone_name, position=None, rotation=None):
        self.model = model
        self.animation_name = animation_name
        self.time = time
        self.bone_name = bone_name
        self.position = position
        self.rotation = rotation
        self.keyframe_index = -1

    def execute(self):
        if self.animation_name in self.model.animations:
            from main import Keyframe  # Import here to avoid circular import
            kf = Keyframe(self.time, self.bone_name, self.position, self.rotation)
            self.model.animations[self.animation_name].keyframes.append(kf)
            self.keyframe_index = len(self.model.animations[self.animation_name].keyframes) - 1
            return True
        return False

    def undo(self):
        if (self.animation_name in self.model.animations and 
            0 <= self.keyframe_index < len(self.model.animations[self.animation_name].keyframes)):
            self.model.animations[self.animation_name].keyframes.pop(self.keyframe_index)

    def description(self):
        return f"Add keyframe to '{self.animation_name}' at time {self.time}"