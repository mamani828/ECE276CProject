def check_node_collision(robot_id, object_ids, joint_indices, joint_position, distance=0.0):
    """
    Checks for:
    1. Environment Collision (Robot vs Cubes)
    2. Self Collision (Left Arm vs Right Arm)
    """
    # 1. Teleport Robot to the configuration
    for j_pos, j_idx in zip(joint_position, joint_indices):
        p.resetJointState(robot_id, j_idx, float(j_pos))
    
    # Force broadphase update so contact checks are accurate
    p.performCollisionDetection()

    # 2. Check Environment Collision
    for object_id in object_ids:
        # Check if robot hits this object
        # Note: We check all links (base + arms) against the object
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=object_id)
        if contacts:
            return True

    # 3. Check Self-Collision (Robot vs Robot)
    # This checks every link against every other link
    self_contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    
    if self_contacts:
        # Optional: Filter out adjacent links if your robot has "false positives" at the elbows.
        # Given your URDF geometry, this simple check should work fine.
        # But if you see collisions always returning True, you can uncomment the filter below:
        
        # for c in self_contacts:
        #     linkA = c[3]
        #     linkB = c[4]
        #     # Ignore adjacent links (parent/child) or same link
        #     if abs(linkA - linkB) > 1: 
        #         return True
        
        return True

    return False