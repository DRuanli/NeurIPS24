from lux.utils import direction_to
import numpy as np
import sys
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class AdvancedAgent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        # Map dimensions
        self.map_width = self.env_cfg["map_width"]
        self.map_height = self.env_cfg["map_height"]

        # Game parameter detection
        self.detected_parameters = {
            "unit_sensor_range": None,
            "nebula_vision_reduction": None,
            "unit_move_cost": None,
            "unit_sap_cost": None,
            "unit_sap_range": None,
        }

        # Memory of the map and objects
        self.relic_nodes_positions = []  # Known relic nodes
        self.discovered_relic_nodes_ids = set()  # IDs of discovered relic nodes
        self.point_yielding_tiles = set()  # (x, y) of tiles that yield points
        self.asteroid_tiles = np.zeros((self.map_width, self.map_height), dtype=bool)  # True if asteroid
        self.nebula_tiles = np.zeros((self.map_width, self.map_height), dtype=bool)  # True if nebula
        self.last_seen_turn = np.zeros((self.map_width, self.map_height), dtype=int)  # Turn when tile was last seen
        self.energy_values = np.zeros((self.map_width, self.map_height), dtype=float)  # Energy at each position

        # Unit management
        self.unit_roles = {}  # unit_id -> role ("explorer", "collector", "defender")
        self.unit_targets = {}  # unit_id -> (x, y) target position
        self.unit_memories = {}  # unit_id -> custom per-unit memory

        # Enemy tracking
        self.last_seen_enemy = np.zeros((self.map_width, self.map_height), dtype=int)  # Turn when enemy was last seen
        self.enemy_unit_positions = []  # Positions of enemy units seen last turn
        self.enemy_unit_energies = {}  # enemy_unit_id -> energy

        # Match tracking
        self.current_match = 0  # Current match number (0-4)
        self.match_wins = 0
        self.match_losses = 0
        self.previous_step = -1  # Last turn processed
        self.turns_with_points = 0  # Number of turns we've scored points

        # Exploration
        self.exploration_targets = []  # (x, y, priority) tuples for exploration
        self.exploration_grid = np.zeros((self.map_width, self.map_height))  # Higher = more interesting to explore

        # Initialization of exploration grid with higher values at center and edges
        x_coords = np.arange(self.map_width)
        y_coords = np.arange(self.map_height)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Center has higher priority
        center_x, center_y = self.map_width // 2, self.map_height // 2
        distance_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        center_priority = 1.0 - distance_from_center / max(self.map_width, self.map_height)

        self.exploration_grid = center_priority.transpose()  # Correcting shape to match (width, height)

    def update_game_parameters(self, obs):
        """Detect and update game parameters based on observations"""
        # Detect unit sensor range if not already detected
        if self.detected_parameters["unit_sensor_range"] is None:
            # Look for the furthest visible empty tile from any unit
            max_visible_distance = 0
            unit_positions = np.array(obs["units"]["position"][self.team_id])
            unit_mask = np.array(obs["units_mask"][self.team_id])
            available_unit_ids = np.where(unit_mask)[0]

            for unit_id in available_unit_ids:
                unit_pos = unit_positions[unit_id]

                # Check all possible visible tiles
                for x in range(self.map_width):
                    for y in range(self.map_height):
                        # If this tile is visible, calculate Manhattan distance
                        dist = abs(x - unit_pos[0]) + abs(y - unit_pos[1])
                        if dist > max_visible_distance:
                            # The unit can see this tile, so update max distance
                            max_visible_distance = dist

            # If we found a visible tile, set sensor range
            if max_visible_distance > 0:
                self.detected_parameters["unit_sensor_range"] = max_visible_distance

        # Detect unit move cost if not already detected
        if self.detected_parameters["unit_move_cost"] is None:
            # If a unit has moved, we can observe the energy reduction
            unit_energies = np.array(obs["units"]["energy"][self.team_id])
            for unit_id in available_unit_ids:
                if unit_id in self.unit_memories and "previous_energy" in self.unit_memories[unit_id]:
                    prev_energy = self.unit_memories[unit_id]["previous_energy"]
                    curr_energy = unit_energies[unit_id]
                    energy_diff = prev_energy - curr_energy

                    # If energy decreased by a value between 1 and 5, it might be the move cost
                    if 1 <= energy_diff <= 5:
                        self.detected_parameters["unit_move_cost"] = energy_diff

        # Update memory of unit energies for next turn
        for unit_id in available_unit_ids:
            if unit_id not in self.unit_memories:
                self.unit_memories[unit_id] = {}
            self.unit_memories[unit_id]["previous_energy"] = unit_energies[unit_id]

    def update_map_knowledge(self, step, obs):
        """Update knowledge about the map based on current observations"""
        # Get player's current vision mask
        vision_mask = np.array(obs["board"]["valid_mask"])

        # Update last seen for visible tiles
        for x in range(self.map_width):
            for y in range(self.map_height):
                if 0 <= x < vision_mask.shape[0] and 0 <= y < vision_mask.shape[1] and vision_mask[x, y]:
                    self.last_seen_turn[x, y] = step

        # Update asteroid and nebula tile knowledge
        # Here, we'd need to actually detect them from the observation, but that depends on the exact format
        # This is just a placeholder for how it might work
        """
        asteroid_mask = np.array(obs["board"]["asteroid_mask"]) 
        nebula_mask = np.array(obs["board"]["nebula_mask"])

        # Update only for visible tiles
        self.asteroid_tiles = np.where(vision_mask, asteroid_mask, self.asteroid_tiles)
        self.nebula_tiles = np.where(vision_mask, nebula_mask, self.nebula_tiles)
        """

        # Track visible relic nodes
        observed_relic_node_positions = np.array(obs["relic_nodes"])
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])
        visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]

        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_nodes_positions.append(observed_relic_node_positions[id])

        # Update point-yielding tiles by checking if our team points increased
        # when we had units on specific tiles near relic nodes
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_mask = np.array(obs["units_mask"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]

        # Track positions of our units
        our_unit_positions = []
        for unit_id in available_unit_ids:
            pos = tuple(unit_positions[unit_id])
            our_unit_positions.append(pos)

            # If we're getting points, record the tiles our units are on
            if step > 0 and "previous_team_points" in self.__dict__:
                current_team_points = obs["team_points"][self.team_id]
                if current_team_points > self.previous_team_points:
                    # We gained points! Record this tile as point-yielding
                    self.point_yielding_tiles.add(pos)
                    self.turns_with_points += 1

        # Track enemy units
        enemy_unit_positions = np.array(obs["units"]["position"][self.opp_team_id])
        enemy_unit_mask = np.array(obs["units_mask"][self.opp_team_id])
        enemy_unit_ids = np.where(enemy_unit_mask)[0]

        # Clear previous enemy positions and update with new ones
        self.enemy_unit_positions = []
        for enemy_id in enemy_unit_ids:
            pos = tuple(enemy_unit_positions[enemy_id])
            self.enemy_unit_positions.append(pos)
            self.last_seen_enemy[pos[0], pos[1]] = step

            # Track enemy energy
            enemy_energy = obs["units"]["energy"][self.opp_team_id][enemy_id]
            self.enemy_unit_energies[enemy_id] = enemy_energy

        # Save the current points for next turn comparison
        self.previous_team_points = obs["team_points"][self.team_id]

        # Update exploration priorities
        self.update_exploration_priorities(step, our_unit_positions)

    def update_exploration_priorities(self, step, our_unit_positions):
        """Update exploration priorities for unexplored areas"""
        # Reduce priority of already visible areas
        decay_rate = 0.95
        exploration_decay = np.ones((self.map_width, self.map_height)) * decay_rate

        # Further decrease priority where our units currently are
        for pos in our_unit_positions:
            x, y = pos
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                exploration_decay[x, y] = 0.5

                # Also decrease nearby tiles
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                            exploration_decay[nx, ny] = 0.7

        # Apply decay to exploration grid
        self.exploration_grid *= exploration_decay

        # Increase priority of areas near relic nodes
        for relic_pos in self.relic_nodes_positions:
            # Create a 5x5 area of higher exploration priority around the relic
            x, y = relic_pos
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                        # Higher priority for cells we haven't confirmed yield points yet
                        if (nx, ny) not in self.point_yielding_tiles:
                            self.exploration_grid[nx, ny] += 0.5

        # Update exploration targets based on exploration grid
        self.exploration_targets = []
        for x in range(self.map_width):
            for y in range(self.map_height):
                if not self.asteroid_tiles[x, y]:  # Don't target asteroid tiles
                    priority = self.exploration_grid[x, y]
                    self.exploration_targets.append((x, y, priority))

        # Sort targets by priority (highest first)
        self.exploration_targets.sort(key=lambda t: t[2], reverse=True)

    def assign_unit_roles(self, step, obs):
        """Assign roles to units based on current strategy and map state"""
        unit_mask = np.array(obs["units_mask"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]

        # Clear old assignments for units that no longer exist
        for unit_id in list(self.unit_roles.keys()):
            if unit_id not in available_unit_ids:
                del self.unit_roles[unit_id]
                if unit_id in self.unit_targets:
                    del self.unit_targets[unit_id]

        # If we haven't found any relic nodes, focus on exploration
        if not self.relic_nodes_positions:
            for unit_id in available_unit_ids:
                self.unit_roles[unit_id] = "explorer"
            return

        # If we found relic nodes but haven't found point-yielding tiles, focus on finding them
        if self.relic_nodes_positions and not self.point_yielding_tiles:
            for unit_id in available_unit_ids:
                self.unit_roles[unit_id] = "explorer"
            return

        # If we know point-yielding tiles, assign most units to collect points
        collector_count = int(len(available_unit_ids) * 0.7)  # 70% collectors
        explorer_count = len(available_unit_ids) - collector_count

        for i, unit_id in enumerate(available_unit_ids):
            if i < collector_count:
                self.unit_roles[unit_id] = "collector"
            else:
                self.unit_roles[unit_id] = "explorer"

        # If we're doing well on points, assign some units to defense
        current_team_points = obs["team_points"][self.team_id]
        opponent_team_points = obs["team_points"][self.opp_team_id]

        if current_team_points > opponent_team_points + 5 and self.turns_with_points > 10:
            # We're ahead by a good margin and have been scoring consistently
            # Convert some explorers to defenders
            defenders_needed = min(2, explorer_count)
            defender_count = 0

            for unit_id in available_unit_ids:
                if self.unit_roles[unit_id] == "explorer" and defender_count < defenders_needed:
                    self.unit_roles[unit_id] = "defender"
                    defender_count += 1

    def assign_unit_targets(self, step, obs):
        """Assign target locations to each unit based on its role"""
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_mask = np.array(obs["units_mask"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]

        for unit_id in available_unit_ids:
            role = self.unit_roles.get(unit_id, "explorer")
            unit_pos = unit_positions[unit_id]

            if role == "explorer":
                # Assign a high-priority exploration target
                for i, (x, y, priority) in enumerate(self.exploration_targets[:10]):  # Consider top 10 targets
                    # Check if this target is already assigned
                    if any(self.unit_targets.get(other_id) == (x, y) for other_id in available_unit_ids if
                           other_id != unit_id):
                        continue

                    # Assign this target
                    self.unit_targets[unit_id] = (x, y)
                    break

                # If no target was assigned, pick a random exploration target
                if unit_id not in self.unit_targets or self.unit_targets[unit_id] is None:
                    if self.exploration_targets:
                        x, y, _ = self.exploration_targets[np.random.randint(0, min(5, len(self.exploration_targets)))]
                        self.unit_targets[unit_id] = (x, y)
                    else:
                        # Fallback: random location
                        x = np.random.randint(0, self.map_width)
                        y = np.random.randint(0, self.map_height)
                        self.unit_targets[unit_id] = (x, y)

            elif role == "collector":
                # Target a point-yielding tile
                if self.point_yielding_tiles:
                    # Find the closest point-yielding tile
                    closest_point_tile = None
                    min_distance = float('inf')

                    for px, py in self.point_yielding_tiles:
                        dist = abs(px - unit_pos[0]) + abs(py - unit_pos[1])
                        if dist < min_distance:
                            min_distance = dist
                            closest_point_tile = (px, py)

                    if closest_point_tile:
                        self.unit_targets[unit_id] = closest_point_tile
                else:
                    # Fallback: target a relic node
                    closest_relic = None
                    min_distance = float('inf')

                    for relic_pos in self.relic_nodes_positions:
                        dist = abs(relic_pos[0] - unit_pos[0]) + abs(relic_pos[1] - unit_pos[1])
                        if dist < min_distance:
                            min_distance = dist
                            closest_relic = relic_pos

                    if closest_relic:
                        # Target a position near the relic node
                        rx, ry = closest_relic
                        offsets = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

                        for dx, dy in offsets:
                            nx, ny = rx + dx, ry + dy
                            if 0 <= nx < self.map_width and 0 <= ny < self.map_height and not self.asteroid_tiles[
                                nx, ny]:
                                self.unit_targets[unit_id] = (nx, ny)
                                break

            elif role == "defender":
                # Target areas with enemy units or protect point-yielding tiles
                if self.enemy_unit_positions:
                    # Find closest enemy
                    closest_enemy = None
                    min_distance = float('inf')

                    for ex, ey in self.enemy_unit_positions:
                        dist = abs(ex - unit_pos[0]) + abs(ey - unit_pos[1])
                        if dist < min_distance:
                            min_distance = dist
                            closest_enemy = (ex, ey)

                    if closest_enemy:
                        self.unit_targets[unit_id] = closest_enemy
                elif self.point_yielding_tiles:
                    # Protect a point-yielding tile
                    if self.point_yielding_tiles:
                        # Pick a random point-yielding tile
                        point_tiles = list(self.point_yielding_tiles)
                        target_tile = point_tiles[np.random.randint(0, len(point_tiles))]
                        self.unit_targets[unit_id] = target_tile

    def determine_actions(self, step, obs):
        """Determine actions for all units based on targets and current state"""
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energies = np.array(obs["units"]["energy"][self.team_id])
        unit_mask = np.array(obs["units_mask"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]

        # Get game params (use detection if available, otherwise default)
        unit_move_cost = self.detected_parameters["unit_move_cost"] or 1
        unit_sap_cost = self.detected_parameters["unit_sap_cost"] or 30
        unit_sap_range = self.detected_parameters["unit_sap_range"] or 4

        # Initialize actions array
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energies[unit_id]
            role = self.unit_roles.get(unit_id, "explorer")
            target = self.unit_targets.get(unit_id)

            # Default to staying in place
            action_direction = 0  # center
            action_sap = 0  # no sap
            action_sap_target = np.zeros(2, dtype=int)  # dummy target

            # Determine if we should move or sap
            should_sap = False
            sap_target = None

            # Check for enemies to sap
            if unit_energy >= unit_sap_cost:
                for enemy_pos in self.enemy_unit_positions:
                    ex, ey = enemy_pos
                    distance = abs(ex - unit_pos[0]) + abs(ey - unit_pos[1])

                    # If enemy is within sap range
                    if distance <= unit_sap_range:
                        should_sap = True
                        sap_target = enemy_pos
                        break

            if should_sap and sap_target:
                # Perform sap action
                action_direction = 5  # sap action
                action_sap = 1  # enabled
                action_sap_target = np.array([
                    sap_target[0] - unit_pos[0],  # delta x
                    sap_target[1] - unit_pos[1]  # delta y
                ])
            elif target and unit_energy >= unit_move_cost:
                # Move toward target
                tx, ty = target

                if (unit_pos[0], unit_pos[1]) == (tx, ty):
                    # Already at target, stay or make small movements
                    if role == "collector" or role == "defender":
                        # Collectors and defenders might want to stay at their target
                        action_direction = 0  # center
                    else:
                        # Explorers should keep moving to discover
                        directions = [1, 2, 3, 4]  # up, right, down, left
                        action_direction = directions[np.random.randint(0, 4)]
                else:
                    # Move toward target
                    action_direction = direction_to(unit_pos, np.array([tx, ty]))
            else:
                # Not enough energy to move or sap, stay in place
                action_direction = 0  # center

            # Set the final action
            actions[unit_id] = [action_direction, action_sap, 0]
            if action_sap:
                # For sap actions, we need to set the target offset
                actions[unit_id, 1:] = action_sap_target

        return actions

    def act(self, step, obs, remainingOverageTime: int = 60):
        """Main function to determine all unit actions for the current turn"""
        # Detect new match
        if step <= self.previous_step:
            self.current_match += 1
        self.previous_step = step

        # Update our knowledge and parameters
        self.update_game_parameters(obs)
        self.update_map_knowledge(step, obs)

        # Assign roles and targets to units
        self.assign_unit_roles(step, obs)
        self.assign_unit_targets(step, obs)

        # Determine and return actions
        actions = self.determine_actions(step, obs)

        return actions


# This wrapper class ensures compatibility with the Lux AI framework
class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.agent = AdvancedAgent(player, env_cfg)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        return self.agent.act(step, obs, remainingOverageTime)