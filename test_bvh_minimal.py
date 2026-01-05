#!/usr/bin/env python3
"""
Minimal test for BVH import
"""

import sys
import os

# Mock the GUI imports to avoid starting the app
class MockQApplication:
    pass

class MockQWidget:
    pass

class MockQMainWindow:
    pass

# Replace the imports
sys.modules['PyQt6.QtWidgets'] = type(sys)('MockQtWidgets')
sys.modules['PyQt6.QtWidgets'].QApplication = MockQApplication
sys.modules['PyQt6.QtWidgets'].QWidget = MockQWidget
sys.modules['PyQt6.QtWidgets'].QMainWindow = MockQMainWindow

# Mock other modules
sys.modules['Graphicalwidget'] = type(sys)('MockGraphical')
sys.modules['autonomous_brain'] = type(sys)('MockBrain')
sys.modules['Creative_conciousness'] = type(sys)('MockCreative')
sys.modules['voice_system'] = type(sys)('MockVoice')
sys.modules['symbolic_math'] = type(sys)('MockMath')
sys.modules['knowledge_base'] = type(sys)('MockKnowledge')
sys.modules['roca.graph_manager'] = type(sys)('MockRoca')

# Now import the Model class
from main import Model, Bone, Animation, Keyframe

def test_bvh_import():
    model = Model()
    
    bvh_file = 'dataset-1_bow_happy_001.bvh'
    
    if not os.path.exists(bvh_file):
        print(f"BVH file {bvh_file} not found")
        return
    
    try:
        count = model.import_bvh(bvh_file)
        print(f"Successfully imported {count} bones")
        print(f"Bones: {list(model.bones.keys())}")
        print(f"Animations: {list(model.animations.keys())}")
        
        # Check if animation has keyframes
        if model.animations:
            anim_name = list(model.animations.keys())[0]
            anim = model.animations[anim_name]
            print(f"Animation '{anim_name}' has {len(anim.keyframes)} keyframes")
            
    except Exception as e:
        print(f"Error importing BVH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bvh_import()