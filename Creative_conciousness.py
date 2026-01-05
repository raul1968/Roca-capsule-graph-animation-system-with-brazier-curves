"""DEPRECATED duplicate file.

This file is a duplicate/misspelled copy of `creative_consciousness.py`.
It has been kept as a placeholder to avoid accidental imports during a
staged refactor. Please import from `creative_consciousness` instead.
"""

import warnings

warnings.warn("Creative_conciousness.py is deprecated; import creative_consciousness instead.")
raise ImportError("Use creative_consciousness.py (canonical) instead of this duplicate file.")
import time

# Import ThoughtType from autonomous_brain
from autonomous_brain import ThoughtType, MentalState

# For natural language generation
try:
    import openai
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    # Fallback to simple templates

# ============================================================================
# MEMORY CONSOLIDATION SYSTEM
# ============================================================================

class ConnectionType(Enum):
    """Types of connections between capsules."""
    VISUAL_SIMILARITY = 1      # Looks similar
    SEMANTIC_RELATION = 2      # Conceptually related
    TEMPORAL_PROXIMITY = 3     # Used around same time
    STYLE_AFFINITY = 4         # Similar artistic style
    EMOTIONAL_TONE = 5         # Similar emotional feel
    NARRATIVE_LINK = 6         # Could be in same story
    TRANSFORMATIONAL = 7       # One could transform into another
    CONTRAST_PAIR = 8          # Interesting contrasts
    METAPHORICAL = 9           # One is metaphor for another
    
@dataclass
class MemoryConnection:
    """A discovered connection between capsules."""
    id: str
    capsule_ids: List[str]
    connection_type: ConnectionType
    strength: float  # 0.0 to 1.0
    description: str
    evidence: List[str] = field(default_factory=list)
    discovered_at: float = field(default_factory=time.time)
    last_recalled: Optional[float] = None
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'capsule_ids': self.capsule_ids,
            'connection_type': self.connection_type.name,
            'strength': self.strength,
            'description': self.description,
            'evidence': self.evidence,
            'discovered_at': self.discovered_at,
            'last_recalled': self.last_recalled,
            'confidence': self.confidence
        }
    
    def recall(self):
        """Mark as recalled (strengthens connection)."""
        self.last_recalled = time.time()
        self.strength = min(1.0, self.strength + 0.05)
        
    def decay(self):
        """Slight decay over time if not recalled."""
        self.strength *= 0.99  # 1% decay

class MemoryConsolidationEngine:
    """
    Finds deep, non-obvious connections between capsules during 'sleep' states.
    Implements creative association and insight generation.
    """
    
    def __init__(self, capsule_manager):
        self.capsule_manager = capsule_manager
        self.connections: Dict[str, MemoryConnection] = {}
        self.connection_graph = nx.Graph()
        self.insights: List[Dict] = []
        
        # Analysis caches
        self.semantic_cache: Dict[str, List[str]] = {}
        self.style_cache: Dict[str, Dict[str, float]] = {}
        self.emotion_cache: Dict[str, List[float]] = {}
        
        # Creativity parameters
        self.association_depth = 3  # How many degrees of separation to explore
        self.similarity_threshold = 0.3
        self.contrast_threshold = 0.7
        
    def deep_sleep_processing(self):
        """
        Main memory consolidation process.
        Called when brain is in DEEP_SLEEP state.
        """
        print("🧠 [Memory Consolidation] Starting deep processing...")
        
        capsules = self.capsule_manager.capsules
        
        if len(capsules) < 3:
            print("🧠 Not enough capsules for deep processing")
            return []
        
        new_connections = []
        
        # Phase 1: Pairwise analysis
        print(f"🧠 Analyzing {len(capsules)} capsules for connections...")
        for i in range(len(capsules)):
            for j in range(i + 1, len(capsules)):
                connections = self._analyze_pair(capsules[i], capsules[j])
                new_connections.extend(connections)
        
        # Phase 2: Higher-order patterns (triads, clusters)
        print("🧠 Looking for higher-order patterns...")
        triad_connections = self._find_triad_patterns(capsules)
        new_connections.extend(triad_connections)
        
        # Phase 3: Narrative weaving
        print("🧠 Weaving narrative connections...")
        narrative_connections = self._weave_narratives(capsules)
        new_connections.extend(narrative_connections)
        
        # Phase 4: Metaphorical connections
        print("🧠 Finding metaphorical links...")
        metaphor_connections = self._find_metaphors(capsules)
        new_connections.extend(metaphor_connections)
        
        # Add new connections
        for conn in new_connections:
            if conn.id not in self.connections:
                self.connections[conn.id] = conn
                # Add to graph
                for cap_id in conn.capsule_ids:
                    self.connection_graph.add_node(cap_id)
                for i in range(len(conn.capsule_ids)):
                    for j in range(i + 1, len(conn.capsule_ids)):
                        self.connection_graph.add_edge(
                            conn.capsule_ids[i], 
                            conn.capsule_ids[j],
                            weight=conn.strength,
                            type=conn.connection_type.name
                        )
        
        # Generate insights from new connections
        insights = self._generate_insights(new_connections)
        
        print(f"🧠 Memory consolidation complete: {len(new_connections)} new connections, {len(insights)} insights")
        return insights
    
    def _analyze_pair(self, cap1, cap2) -> List[MemoryConnection]:
        """Analyze a pair of capsules for various connection types."""
        connections = []
        
        # 1. Name-based semantic connections
        if self._semantic_connection_score(cap1, cap2) > 0.6:
            conn = MemoryConnection(
                id=f"semantic_{cap1.uuid[:8]}_{cap2.uuid[:8]}",
                capsule_ids=[cap1.uuid, cap2.uuid],
                connection_type=ConnectionType.SEMANTIC_RELATION,
                strength=self._semantic_connection_score(cap1, cap2),
                description=f"'{cap1.name}' and '{cap2.name}' are conceptually related",
                evidence=[f"Name similarity: {cap1.name} ↔ {cap2.name}"]
            )
            connections.append(conn)
        
        # 2. Metadata analysis
        common_metadata = self._common_metadata(cap1, cap2)
        if common_metadata:
            conn = MemoryConnection(
                id=f"meta_{cap1.uuid[:8]}_{cap2.uuid[:8]}",
                capsule_ids=[cap1.uuid, cap2.uuid],
                connection_type=ConnectionType.SEMANTIC_RELATION,
                strength=len(common_metadata) * 0.2,
                description=f"'{cap1.name}' and '{cap2.name}' share metadata",
                evidence=[f"Shared metadata: {', '.join(common_metadata)}"]
            )
            connections.append(conn)
        
        # 3. Style analysis (if images available)
        style_sim = self._style_similarity(cap1, cap2)
        if style_sim > self.similarity_threshold:
            conn = MemoryConnection(
                id=f"style_{cap1.uuid[:8]}_{cap2.uuid[:8]}",
                capsule_ids=[cap1.uuid, cap2.uuid],
                connection_type=ConnectionType.STYLE_AFFINITY,
                strength=style_sim,
                description=f"'{cap1.name}' and '{cap2.name}' have similar artistic style",
                evidence=[f"Style similarity: {style_sim:.2f}"]
            )
            connections.append(conn)
        
        # 4. Temporal proximity (usage patterns)
        time_diff = abs(cap1.last_used_time - cap2.last_used_time)
        if time_diff < 3600:  # Used within same hour
            conn = MemoryConnection(
                id=f"temporal_{cap1.uuid[:8]}_{cap2.uuid[:8]}",
                capsule_ids=[cap1.uuid, cap2.uuid],
                connection_type=ConnectionType.TEMPORAL_PROXIMITY,
                strength=max(0, 1.0 - time_diff / 3600),
                description=f"'{cap1.name}' and '{cap2.name}' are often used together",
                evidence=[f"Used within {int(time_diff/60)} minutes of each other"]
            )
            connections.append(conn)
        
        # 5. Contrast analysis (interesting opposites)
        contrast_score = self._contrast_score(cap1, cap2)
        if contrast_score > self.contrast_threshold:
            conn = MemoryConnection(
                id=f"contrast_{cap1.uuid[:8]}_{cap2.uuid[:8]}",
                capsule_ids=[cap1.uuid, cap2.uuid],
                connection_type=ConnectionType.CONTRAST_PAIR,
                strength=contrast_score,
                description=f"'{cap1.name}' and '{cap2.name}' form an interesting contrast",
                evidence=[f"Contrast score: {contrast_score:.2f}"]
            )
            connections.append(conn)
        
        return connections
    
    def _find_triad_patterns(self, capsules) -> List[MemoryConnection]:
        """Find patterns among three capsules."""
        connections = []
        
        if len(capsules) < 3:
            return connections
        
        # Look for thematic triads
        for i in range(len(capsules)):
            for j in range(i + 1, len(capsules)):
                for k in range(j + 1, len(capsules)):
                    cap1, cap2, cap3 = capsules[i], capsules[j], capsules[k]
                    
                    # Check if they form a progression
                    if self._forms_progression([cap1, cap2, cap3]):
                        conn = MemoryConnection(
                            id=f"progression_{cap1.uuid[:8]}_{cap2.uuid[:8]}_{cap3.uuid[:8]}",
                            capsule_ids=[cap1.uuid, cap2.uuid, cap3.uuid],
                            connection_type=ConnectionType.TRANSFORMATIONAL,
                            strength=0.7,
                            description=f"'{cap1.name}' → '{cap2.name}' → '{cap3.name}' shows a clear progression",
                            evidence=["Forms a logical sequence or transformation"]
                        )
                        connections.append(conn)
                    
                    # Check for complementary set
                    if self._forms_complementary_set([cap1, cap2, cap3]):
                        conn = MemoryConnection(
                            id=f"complement_{cap1.uuid[:8]}_{cap2.uuid[:8]}_{cap3.uuid[:8]}",
                            capsule_ids=[cap1.uuid, cap2.uuid, cap3.uuid],
                            connection_type=ConnectionType.SEMANTIC_RELATION,
                            strength=0.6,
                            description=f"'{cap1.name}', '{cap2.name}', and '{cap3.name}' complement each other",
                            evidence=["Forms a complete or complementary set"]
                        )
                        connections.append(conn)
        
        return connections
    
    def _weave_narratives(self, capsules) -> List[MemoryConnection]:
        """Create narrative connections between capsules."""
        connections = []
        
        # Group by type for narrative construction
        characters = [c for c in capsules if c.type.lower() == "character"]
        poses = [c for c in capsules if c.type.lower() == "pose"]
        others = [c for c in capsules if c.type.lower() not in ["character", "pose"]]
        
        # Create character-pose narratives
        for char in characters[:3]:  # Limit to first 3 characters
            if poses:
                # Find poses that might fit this character
                fitting_poses = self._find_fitting_poses(char, poses)
                
                for pose in fitting_poses[:2]:  # Limit to 2 poses per character
                    conn = MemoryConnection(
                        id=f"narrative_{char.uuid[:8]}_{pose.uuid[:8]}",
                        capsule_ids=[char.uuid, pose.uuid],
                        connection_type=ConnectionType.NARRATIVE_LINK,
                        strength=0.8,
                        description=f"'{char.name}' could be doing the '{pose.name}' pose in a story",
                        evidence=[f"Character-pose compatibility"]
                    )
                    connections.append(conn)
        
        # Create mini-scenes (3-4 capsules that could be a scene)
        if len(capsules) >= 4:
            for _ in range(min(5, len(capsules) // 4)):  # Try up to 5 scenes
                scene_capsules = random.sample(capsules, 4)
                if self._could_be_scene(scene_capsules):
                    conn = MemoryConnection(
                        id=f"scene_{'_'.join(c.uuid[:8] for c in scene_capsules)}",
                        capsule_ids=[c.uuid for c in scene_capsules],
                        connection_type=ConnectionType.NARRATIVE_LINK,
                        strength=0.6,
                        description=f"These 4 capsules could form an interesting scene",
                        evidence=["Forms a coherent potential scene"]
                    )
                    connections.append(conn)
        
        return connections
    
    def _find_metaphors(self, capsules) -> List[MemoryConnection]:
        """Find metaphorical connections between capsules."""
        connections = []
        
        # Look for abstract-concept capsules that could metaphorize others
        abstract_capsules = [c for c in capsules if self._is_abstract(c)]
        concrete_capsules = [c for c in capsules if not self._is_abstract(c)]
        
        for abstract in abstract_capsules[:3]:  # Limit
            for concrete in concrete_capsules[:5]:  # Limit
                metaphor_strength = self._metaphor_strength(abstract, concrete)
                if metaphor_strength > 0.6:
                    conn = MemoryConnection(
                        id=f"metaphor_{abstract.uuid[:8]}_{concrete.uuid[:8]}",
                        capsule_ids=[abstract.uuid, concrete.uuid],
                        connection_type=ConnectionType.METAPHORICAL,
                        strength=metaphor_strength,
                        description=f"'{concrete.name}' could be a metaphor for '{abstract.name}'",
                        evidence=[f"Metaphorical connection strength: {metaphor_strength:.2f}"]
                    )
                    connections.append(conn)
        
        return connections
    
    def _generate_insights(self, new_connections: List[MemoryConnection]) -> List[Dict]:
        """Generate insights from discovered connections."""
        insights = []
        
        # Group connections by type
        by_type = defaultdict(list)
        for conn in new_connections:
            by_type[conn.connection_type].append(conn)
        
        # Generate insights for each connection type
        for conn_type, conns in by_type.items():
            if len(conns) >= 2:
                insight = self._create_insight(conn_type, conns)
                if insight:
                    insights.append(insight)
        
        # Look for network clusters
        if len(new_connections) >= 5:
            cluster_insight = self._find_cluster_insights(new_connections)
            if cluster_insight:
                insights.append(cluster_insight)
        
        # Look for surprising connections
        surprising = [c for c in new_connections 
                     if c.connection_type in [ConnectionType.METAPHORICAL, 
                                            ConnectionType.CONTRAST_PAIR,
                                            ConnectionType.TRANSFORMATIONAL]]
        if surprising:
            surprise_insight = {
                'type': 'surprising_connections',
                'title': 'Surprising Discoveries',
                'content': f"Found {len(surprising)} unexpected connections between seemingly unrelated capsules",
                'connections': [c.to_dict() for c in surprising[:3]],
                'priority': 0.8
            }
            insights.append(surprise_insight)
        
        return insights
    
    def _create_insight(self, conn_type: ConnectionType, connections: List[MemoryConnection]) -> Optional[Dict]:
        """Create insight text for a group of connections."""
        if not connections:
            return None
        
        # Get capsules involved
        all_capsules = []
        for conn in connections:
            for cap_id in conn.capsule_ids:
                capsule = self.capsule_manager.get_capsule_by_uuid(cap_id)
                if capsule:
                    all_capsules.append(capsule)
        
        # Unique capsule names
        unique_names = list(set(c.name for c in all_capsules))
        
        # Create insight based on type
        if conn_type == ConnectionType.SEMANTIC_RELATION:
            return {
                'type': 'semantic_cluster',
                'title': 'Thematic Group Discovered',
                'content': f"Found semantic connections between {len(unique_names)} capsules: {', '.join(unique_names[:5])}",
                'priority': 0.6
            }
        elif conn_type == ConnectionType.NARRATIVE_LINK:
            return {
                'type': 'narrative_potential',
                'title': 'Story Potential',
                'content': f"These {len(unique_names)} capsules could form interesting narratives together",
                'priority': 0.7
            }
        elif conn_type == ConnectionType.STYLE_AFFINITY:
            return {
                'type': 'style_cluster',
                'title': 'Style Consistency',
                'content': f"Found {len(connections)} style similarities, suggesting a consistent artistic approach",
                'priority': 0.5
            }
        
        return None
    
    def _find_cluster_insights(self, connections: List[MemoryConnection]) -> Optional[Dict]:
        """Find insights from connection clusters in the graph."""
        if not self.connection_graph:
            return None
        
        # Find connected components
        components = list(nx.connected_components(self.connection_graph))
        
        if len(components) > 1:
            largest = max(components, key=len)
            
            # Get capsule names in largest component
            capsule_names = []
            for cap_id in list(largest)[:5]:  # First 5
                capsule = self.capsule_manager.get_capsule_by_uuid(cap_id)
                if capsule:
                    capsule_names.append(capsule.name)
            
            if capsule_names:
                return {
                    'type': 'connection_cluster',
                    'title': 'Highly Connected Group',
                    'content': f"Found a cluster of {len(largest)} tightly connected capsules including: {', '.join(capsule_names)}",
                    'priority': 0.7
                }
        
        return None
    
    # ============================================================================
    # ANALYSIS HELPER METHODS
    # ============================================================================
    
    def _semantic_connection_score(self, cap1, cap2) -> float:
        """Calculate semantic connection score based on names and metadata."""
        score = 0.0
        
        # Name similarity
        name1 = cap1.name.lower()
        name2 = cap2.name.lower()
        
        # Check for common words
        words1 = set(name1.split())
        words2 = set(name2.split())
        common_words = words1 & words2
        
        if common_words:
            score += len(common_words) * 0.3
        
        # Check for substring matches
        if name1 in name2 or name2 in name1:
            score += 0.4
        
        # Metadata similarity
        meta1 = str(cap1.metadata).lower()
        meta2 = str(cap2.metadata).lower()
        
        # Simple word overlap in metadata
        meta_words1 = set(re.findall(r'\w+', meta1))
        meta_words2 = set(re.findall(r'\w+', meta2))
        common_meta = meta_words1 & meta_words2
        
        if common_meta:
            score += len(common_meta) * 0.1
        
        return min(1.0, score)
    
    def _common_metadata(self, cap1, cap2) -> List[str]:
        """Find common metadata keys or values."""
        common = []
        
        if not isinstance(cap1.metadata, dict) or not isinstance(cap2.metadata, dict):
            return common
        
        # Common keys
        common_keys = set(cap1.metadata.keys()) & set(cap2.metadata.keys())
        common.extend(list(common_keys))
        
        # Common values (for overlapping keys)
        for key in common_keys:
            if cap1.metadata[key] == cap2.metadata[key]:
                common.append(f"{key}:{cap1.metadata[key]}")
        
        return common
    
    def _style_similarity(self, cap1, cap2) -> float:
        """Calculate style similarity (placeholder implementation)."""
        # In reality, this would analyze image features
        # For now, use metadata heuristics
        
        score = 0.0
        
        # Check for style-related metadata
        style_keys = ['style', 'palette', 'line_quality', 'rendering']
        
        for key in style_keys:
            val1 = cap1.metadata.get(key, '').lower() if isinstance(cap1.metadata, dict) else ''
            val2 = cap2.metadata.get(key, '').lower() if isinstance(cap2.metadata, dict) else ''
            
            if val1 and val2 and val1 == val2:
                score += 0.3
        
        # Type similarity
        if cap1.type == cap2.type:
            score += 0.2
        
        return min(1.0, score)
    
    def _contrast_score(self, cap1, cap2) -> float:
        """Calculate interesting contrast score."""
        score = 0.0
        
        # Type contrast
        if cap1.type != cap2.type:
            score += 0.3
        
        # Name length contrast
        len_diff = abs(len(cap1.name) - len(cap2.name))
        if len_diff > 5:
            score += 0.2
        
        # Conceptual opposites (simple heuristic)
        opposites = [
            ('light', 'dark'), ('big', 'small'), ('fast', 'slow'),
            ('happy', 'sad'), ('simple', 'complex'), ('old', 'new')
        ]
        
        name1 = cap1.name.lower()
        name2 = cap2.name.lower()
        
        for opp1, opp2 in opposites:
            if (opp1 in name1 and opp2 in name2) or (opp2 in name1 and opp1 in name2):
                score += 0.5
        
        return min(1.0, score)
    
    def _forms_progression(self, capsules) -> bool:
        """Check if capsules form a logical progression."""
        # Sort by name length (simple proxy for complexity)
        sorted_by_len = sorted(capsules, key=lambda c: len(c.name))
        
        # Check if names suggest progression
        progression_indicators = ['1', '2', '3', 'first', 'second', 'third', 
                                 'beginning', 'middle', 'end', 'start', 'finish']
        
        indicator_count = sum(1 for cap in capsules 
                            for ind in progression_indicators 
                            if ind in cap.name.lower())
        
        return indicator_count >= 2
    
    def _forms_complementary_set(self, capsules) -> bool:
        """Check if capsules form a complementary set."""
        # Look for sets like: morning, afternoon, evening
        # or: pencil, pen, brush
        
        # Group by first letter or prefix
        prefixes = defaultdict(list)
        for cap in capsules:
            prefix = cap.name[:3].lower()
            prefixes[prefix].append(cap)
        
        # Check for common prefix with variations
        for prefix, caps in prefixes.items():
            if len(caps) >= 3:
                return True
        
        return False
    
    def _find_fitting_poses(self, character, poses) -> List:
        """Find poses that fit a character."""
        fitting = []
        
        char_name = character.name.lower()
        
        for pose in poses:
            pose_name = pose.name.lower()
            
            # Simple heuristic: if pose name contains character name
            # or character type matches pose metadata
            if char_name in pose_name:
                fitting.append(pose)
            elif isinstance(character.metadata, dict) and isinstance(pose.metadata, dict):
                char_type = character.metadata.get('character_type', '').lower()
                pose_for = pose.metadata.get('for_character', '').lower()
                
                if char_type and pose_for and char_type in pose_for:
                    fitting.append(pose)
        
        return fitting
    
    def _could_be_scene(self, capsules) -> bool:
        """Check if capsules could form a coherent scene."""
        # Need at least one character
        characters = [c for c in capsules if c.type.lower() == "character"]
        if not characters:
            return False
        
        # Need at least one pose or action
        actions = [c for c in capsules if c.type.lower() in ["pose", "skill"]]
        if not actions:
            return False
        
        # Diversity check (not all same type)
        types = set(c.type.lower() for c in capsules)
        return len(types) >= 2
    
    def _is_abstract(self, capsule) -> bool:
        """Check if capsule represents an abstract concept."""
        abstract_indicators = ['emotion', 'concept', 'idea', 'theme', 
                              'style', 'mood', 'energy', 'spirit']
        
        name = capsule.name.lower()
        for indicator in abstract_indicators:
            if indicator in name:
                return True
        
        # Check metadata
        if isinstance(capsule.metadata, dict):
            description = str(capsule.metadata.get('description', '')).lower()
            for indicator in abstract_indicators:
                if indicator in description:
                    return True
        
        return False
    
    def _metaphor_strength(self, abstract, concrete) -> float:
        """Calculate metaphorical connection strength."""
        score = 0.0
        
        # Name-based metaphor detection
        abstract_name = abstract.name.lower()
        concrete_name = concrete.name.lower()
        
        # Common metaphorical mappings
        metaphors = {
            'light': ['sun', 'lamp', 'star', 'fire'],
            'dark': ['shadow', 'night', 'cave', 'void'],
            'anger': ['fire', 'storm', 'volcano', 'beast'],
            'joy': ['sunshine', 'flower', 'song', 'dance'],
            'time': ['river', 'clock', 'sand', 'wheel'],
            'love': ['heart', 'rose', 'embrace', 'warmth']
        }
        
        # Check if abstract concept maps to concrete object
        for concept, objects in metaphors.items():
            if concept in abstract_name:
                for obj in objects:
                    if obj in concrete_name:
                        score += 0.5
        
        # Check metadata for metaphorical connections
        if isinstance(abstract.metadata, dict) and isinstance(concrete.metadata, dict):
            abstract_desc = str(abstract.metadata.get('description', '')).lower()
            concrete_desc = str(concrete.metadata.get('description', '')).lower()
            
            # Simple word overlap in descriptive language
            abstract_words = set(re.findall(r'\w+', abstract_desc))
            concrete_words = set(re.findall(r'\w+', concrete_desc))
            
            if abstract_words & concrete_words:
                score += 0.3
        
        return min(1.0, score)
    
    def get_suggested_connections(self, capsule_id: str, limit: int = 5) -> List[Dict]:
        """Get suggested connections for a specific capsule."""
        suggestions = []
        
        if capsule_id not in self.connection_graph:
            return suggestions
        
        # Get connected capsules
        if capsule_id in self.connection_graph:
            neighbors = list(self.connection_graph.neighbors(capsule_id))
            
            for neighbor_id in neighbors[:limit]:
                # Find connection between these capsules
                for conn in self.connections.values():
                    if (capsule_id in conn.capsule_ids and 
                        neighbor_id in conn.capsule_ids):
                        
                        capsule = self.capsule_manager.get_capsule_by_uuid(capsule_id)
                        neighbor = self.capsule_manager.get_capsule_by_uuid(neighbor_id)
                        
                        if capsule and neighbor:
                            suggestions.append({
                                'capsule1': capsule.name,
                                'capsule2': neighbor.name,
                                'connection': conn.description,
                                'type': conn.connection_type.name,
                                'strength': conn.strength
                            })
        
        return suggestions
    
    def get_creative_prompts(self, count: int = 3) -> List[str]:
        """Generate creative prompts based on connections."""
        prompts = []
        
        # Get some interesting connections
        interesting = sorted(self.connections.values(), 
                           key=lambda c: c.strength * (c.confidence or 1.0), 
                           reverse=True)[:10]
        
        for conn in interesting[:count]:
            capsules = []
            for cap_id in conn.capsule_ids:
                cap = self.capsule_manager.get_capsule_by_uuid(cap_id)
                if cap:
                    capsules.append(cap)
            
            if len(capsules) >= 2:
                if conn.connection_type == ConnectionType.METAPHORICAL:
                    prompts.append(
                        f"Create an artwork where '{capsules[0].name}' represents '{capsules[1].name}' metaphorically"
                    )
                elif conn.connection_type == ConnectionType.NARRATIVE_LINK:
                    prompts.append(
                        f"Tell a story that includes both '{capsules[0].name}' and '{capsules[1].name}'"
                    )
                elif conn.connection_type == ConnectionType.TRANSFORMATIONAL:
                    prompts.append(
                        f"Animate a transformation from '{capsules[0].name}' to '{capsules[1].name}'"
                    )
                elif conn.connection_type == ConnectionType.CONTRAST_PAIR:
                    prompts.append(
                        f"Explore the contrast between '{capsules[0].name}' and '{capsules[1].name}' in a single composition"
                    )
        
        return prompts

# ============================================================================
# TEACHING MODULE
# ============================================================================

class AnimationPrinciple(Enum):
    """Animation principles to teach."""
    SQUASH_AND_STRETCH = 1
    ANTICIPATION = 2
    STAGING = 3
    STRAIGHT_AHEAD = 4
    FOLLOW_THROUGH = 5
    SLOW_IN_SLOW_OUT = 6
    ARCS = 7
    SECONDARY_ACTION = 8
    TIMING = 9
    EXAGGERATION = 10
    SOLID_DRAWING = 11
    APPEAL = 12

@dataclass
class TeachingMoment:
    """A teaching moment opportunity."""
    principle: AnimationPrinciple
    context: str  # What the user is doing
    difficulty_level: float  # 0.0 to 1.0
    estimated_time: int  # Seconds
    examples: List[str] = field(default_factory=list)
    interactive_exercise: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'principle': self.principle.name,
            'context': self.context,
            'difficulty': self.difficulty_level,
            'time': self.estimated_time,
            'examples': self.examples,
            'has_exercise': self.interactive_exercise is not None
        }

class TeachingEngine:
    """
    Adaptive animation teaching system.
    Detects when user is struggling and provides helpful lessons.
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.user_skill_level = 0.5  # Track user's skill
        self.taught_principles: Set[AnimationPrinciple] = set()
        self.struggle_indicators = defaultdict(int)
        
        # Lesson database
        self.lessons = self._initialize_lessons()
        
        # Learning progress tracking
        self.learning_progress = defaultdict(float)  # principle -> progress (0-1)
        
    def _initialize_lessons(self) -> Dict[AnimationPrinciple, Dict]:
        """Initialize animation principle lessons."""
        return {
            AnimationPrinciple.SQUASH_AND_STRETCH: {
                'title': 'Squash and Stretch',
                'description': 'Giving weight and flexibility to objects',
                'difficulty': 0.3,
                'key_points': [
                    'Shows the rigidity of material',
                    'Volume should remain constant',
                    'Exaggerate for cartoony style',
                    'Subtle for realistic animation'
                ],
                'common_mistakes': [
                    'Changing volume during squash/stretch',
                    'Overusing the effect',
                    'Inconsistent application'
                ],
                'examples': [
                    'Bouncing ball: stretches when falling, squashes on impact',
                    'Character jumping: stretches up, squashes landing'
                ]
            },
            AnimationPrinciple.ANTICIPATION: {
                'title': 'Anticipation',
                'description': 'Preparing the audience for an action',
                'difficulty': 0.4,
                'key_points': [
                    'Creates expectation',
                    'Makes actions feel more natural',
                    'Timing depends on action weight',
                    'Can be subtle or exaggerated'
                ],
                'common_mistakes': [
                    'Skipping anticipation makes actions feel robotic',
                    'Too much anticipation slows down animation',
                    'Inconsistent anticipation timing'
                ],
                'examples': [
                    'Before throwing: character winds up',
                    'Before jumping: character crouches down',
                    'Before speaking: character inhales'
                ]
            },
            AnimationPrinciple.TIMING: {
                'title': 'Timing and Spacing',
                'description': 'Controlling the speed and rhythm of movement',
                'difficulty': 0.6,
                'key_points': [
                    'More frames = slower movement',
                    'Closer spacing = slower movement',
                    'Timing creates weight and personality',
                    'Vary timing for natural feel'
                ],
                'common_mistakes': [
                    'Even spacing (robot movement)',
                    'Inconsistent timing between actions',
                    'Not matching timing to character weight'
                ],
                'examples': [
                    'Heavy character: slow starts and stops',
                    'Light character: quick, bouncy movements',
                    'Easing: slow in/slow out for natural motion'
                ]
            },
            AnimationPrinciple.ARCS: {
                'title': 'Arcs',
                'description': 'Natural movement follows curved paths',
                'difficulty': 0.5,
                'key_points': [
                    'Most natural movement follows arcs',
                    'Straight line movement looks mechanical',
                    'Different arcs for different actions',
                    'Use reference lines to plan arcs'
                ],
                'common_mistakes': [
                    'Straight line arm swings',
                    'Inconsistent arc paths',
                    'Not following through on arcs'
                ],
                'examples': [
                    'Arm swing: curved path from shoulder',
                    'Head turn: slight arc, not straight pivot',
                    'Walk cycle: up-down arc of body'
                ]
            }
        }
    
    def analyze_for_teaching_moments(self) -> Optional[TeachingMoment]:
        """
        Analyze current context for teaching opportunities.
        Returns a teaching moment if appropriate.
        """
        # Check if user might be struggling
        struggle_level = self._detect_struggle()
        
        # Don't interrupt if user is in flow
        if self.brain.state in [MentalState.CREATING, MentalState.ATTENTIVE]:
            if struggle_level < 0.6:  # Only interrupt for significant struggle
                return None
        
        # Don't teach too frequently
        if time.time() - self._last_teaching_time() < 300:  # 5 minutes minimum
            return None
        
        # Find appropriate principle to teach
        principle = self._select_teaching_principle()
        if not principle:
            return None
        
        # Create teaching moment
        context = self._get_current_context()
        
        return TeachingMoment(
            principle=principle,
            context=context,
            difficulty_level=self.lessons[principle]['difficulty'],
            estimated_time=120,  # 2 minutes average
            examples=self.lessons[principle]['examples'][:2],
            interactive_exercise=self._create_exercise(principle)
        )
    
    def _detect_struggle(self) -> float:
        """Detect if user is struggling with animation."""
        struggle_score = 0.0
        
        # Analyze recent brain thoughts for struggle indicators
        recent_thoughts = list(self.brain.thought_stream)[-10:]
        
        # Look for struggle indicators in thoughts
        struggle_indicators = [
            'struggling', 'difficult', 'hard', 'confused', 'stuck',
            'fix', 'problem', 'issue', 'help', 'how to'
        ]
        
        for thought in recent_thoughts:
            content_lower = thought.content.lower()
            for indicator in struggle_indicators:
                if indicator in content_lower:
                    struggle_score += 0.1
        
        # Check for repetitive actions (might indicate struggle)
        if len(recent_thoughts) >= 3:
            # Count similar thoughts
            thought_types = [t.type for t in recent_thoughts]
            type_counts = defaultdict(int)
            for t_type in thought_types:
                type_counts[t_type] += 1
            
            # High count of OBSERVATION might mean user is stuck observing
            if type_counts.get(ThoughtType.OBSERVATION, 0) >= 5:
                struggle_score += 0.3
        
        # Check timeline for struggle patterns
        if hasattr(self.brain, 'timeline'):
            timeline = self.brain.timeline
            if timeline and timeline.animation:
                frames = timeline.animation.frames
                
                # Many similar frames might indicate struggle
                if len(frames) >= 3:
                    # Simple heuristic: frames added rapidly then deleted
                    # (In reality, would need more sophisticated analysis)
                    struggle_score += 0.1
        
        return min(1.0, struggle_score)
    
    def _select_teaching_principle(self) -> Optional[AnimationPrinciple]:
        """Select which principle to teach based on context."""
        # Get principles not yet taught
        available = [p for p in AnimationPrinciple 
                    if p not in self.taught_principles]
        
        if not available:
            # Re-teach principles with low progress
            available = [p for p in AnimationPrinciple 
                        if self.learning_progress[p] < 0.7]
        
        if not available:
            return None
        
        # Match principle to current activity
        context = self._get_current_context()
        
        # Map context to principles
        context_to_principles = {
            'drawing_poses': [AnimationPrinciple.SQUASH_AND_STRETCH, 
                            AnimationPrinciple.EXAGGERATION],
            'creating_animation': [AnimationPrinciple.TIMING, 
                                 AnimationPrinciple.ANTICIPATION,
                                 AnimationPrinciple.FOLLOW_THROUGH],
            'character_design': [AnimationPrinciple.APPEAL, 
                               AnimationPrinciple.SOLID_DRAWING],
            'motion_issues': [AnimationPrinciple.ARCS, 
                            AnimationPrinciple.SLOW_IN_SLOW_OUT]
        }
        
        # Get relevant principles for current context
        relevant = []
        for ctx_key, principles in context_to_principles.items():
            if ctx_key in context:
                relevant.extend(principles)
        
        # Filter to available principles
        relevant = [p for p in relevant if p in available]
        
        if relevant:
            # Choose based on user skill level
            suitable = [p for p in relevant 
                       if self.lessons[p]['difficulty'] <= self.user_skill_level + 0.2]
            
            if suitable:
                return random.choice(suitable)
            elif relevant:
                return random.choice(relevant)
        
        # Fallback: random available principle
        return random.choice(available)
    
    def _get_current_context(self) -> str:
        """Get current user context for teaching."""
        context_parts = []
        
        # Check what user is focused on
        if self.brain.attention_focus == "canvas":
            if self.brain.canvas and self.brain.canvas.drawing:
                context_parts.append("drawing_poses")
        
        # Check timeline activity
        if self.brain.timeline and self.brain.timeline.animation:
            frame_count = self.brain.timeline.animation.get_frame_count()
            if frame_count > 1:
                context_parts.append("creating_animation")
        
        # Check capsule activity
        capsules = self.brain.capsule_manager.capsules
        characters = [c for c in capsules if c.type.lower() == "character"]
        if characters:
            context_parts.append("character_design")
        
        # Check for motion-related thoughts
        recent_thoughts = list(self.brain.thought_stream)[-5:]
        motion_words = ['move', 'motion', 'animate', 'timing', 'speed', 'flow']
        for thought in recent_thoughts:
            if any(word in thought.content.lower() for word in motion_words):
                context_parts.append("motion_issues")
                break
        
        return "_".join(context_parts) if context_parts else "general"
    
    def _create_exercise(self, principle: AnimationPrinciple) -> Optional[Dict]:
        """Create an interactive exercise for the principle."""
        exercises = {
            AnimationPrinciple.SQUASH_AND_STRETCH: {
                'type': 'drawing_exercise',
                'title': 'Bouncing Ball Challenge',
                'instructions': 'Draw a ball bouncing with squash on impact and stretch during fall',
                'steps': [
                    'Draw ball at top position (normal shape)',
                    'Draw falling ball (slightly stretched downward)',
                    'Draw impact frame (squashed horizontally)',
                    'Draw rebounding ball (stretched upward)'
                ],
                'hints': [
                    'Keep volume consistent',
                    'Exaggerate for cartoony effect',
                    'Follow through with bounce decay'
                ]
            },
            AnimationPrinciple.ANTICIPATION: {
                'type': 'animation_exercise',
                'title': 'Anticipation Practice',
                'instructions': 'Add anticipation to this action: character throwing a ball',
                'steps': [
                    'Start with character in neutral pose',
                    'Add wind-up pose (anticipation)',
                    'Show throwing action',
                    'Add follow-through'
                ],
                'hints': [
                    'Anticipation should be opposite direction of main action',
                    'Timing: anticipation slower than action',
                    'Exaggerate for clarity'
                ]
            },
            AnimationPrinciple.TIMING: {
                'type': 'timing_exercise',
                'title': 'Timing and Spacing Practice',
                'instructions': 'Create a pendulum swing with proper timing',
                'steps': [
                    'Draw pendulum at left extreme',
                    'Add middle position (fastest)',
                    'Draw right extreme position',
                    'Add slow-in/slow-out spacing'
                ],
                'hints': [
                    'Closer spacing at extremes',
                    'Wider spacing in middle',
                    'Use timing chart: [1, 3, 5, 7]'
                ]
            },
            AnimationPrinciple.ARCS: {
                'type': 'drawing_exercise',
                'title': 'Arc Movement Practice',
                'instructions': 'Draw a character waving hand in an arc',
                'steps': [
                    'Mark arc path with light guidelines',
                    'Draw key poses along arc',
                    'Add in-between poses following arc',
                    'Clean up final animation'
                ],
                'hints': [
                    'Use circular guides',
                    'Wrist leads the movement',
                    'Follow through beyond main arc'
                ]
            }
        }
        
        return exercises.get(principle)
    
    def _last_teaching_time(self) -> float:
        """Get time of last teaching moment."""
        # In full implementation, would track actual teaching moments
        # For now, return a timestamp far in the past
        return time.time() - 1000  # 1000 seconds ago
    
    def deliver_lesson(self, teaching_moment: TeachingMoment) -> Dict:
        """Deliver a lesson to the user."""
        principle = teaching_moment.principle
        lesson_data = self.lessons[principle]
        
        # Format lesson
        lesson = {
            'title': f"Animation Principle: {lesson_data['title']}",
            'principle': principle.name,
            'description': lesson_data['description'],
            'context': f"Since you're {teaching_moment.context.replace('_', ' ')}, this might help:",
            'key_points': lesson_data['key_points'],
            'examples': teaching_moment.examples,
            'common_mistakes': lesson_data.get('common_mistakes', []),
            'interactive_exercise': teaching_moment.interactive_exercise,
            'estimated_time': teaching_moment.estimated_time
        }
        
        # Mark as taught
        self.taught_principles.add(principle)
        
        # Update learning progress
        self.learning_progress[principle] = 0.3  # Initial learning
        
        return lesson
    
    def record_exercise_completion(self, principle: AnimationPrinciple, quality: float):
        """Record completion of an exercise."""
        # Update learning progress
        current = self.learning_progress.get(principle, 0.0)
        self.learning_progress[principle] = min(1.0, current + quality * 0.3)
        
        # Update user skill level
        self.user_skill_level = min(1.0, self.user_skill_level + quality * 0.05)
        
        # Provide feedback based on quality
        if quality > 0.8:
            return "Excellent work! You've really grasped this principle."
        elif quality > 0.6:
            return "Good job! With more practice, you'll master this."
        else:
            return "Keep practicing! Animation takes time to learn."
    
    def get_learning_report(self) -> Dict:
        """Get report on user's learning progress."""
        return {
            'skill_level': self.user_skill_level,
            'principles_learned': len(self.taught_principles),
            'progress_details': {
                p.name: self.learning_progress.get(p, 0.0)
                for p in AnimationPrinciple
            },
            'next_recommended': self._get_next_recommendations()
        }
    
    def _get_next_recommendations(self) -> List[Dict]:
        """Get recommendations for next learning steps."""
        recommendations = []
        
        # Principles with low progress
        for principle in AnimationPrinciple:
            progress = self.learning_progress.get(principle, 0.0)
            if progress < 0.5 and principle not in self.taught_principles:
                recommendations.append({
                    'principle': principle.name,
                    'reason': 'Not yet learned',
                    'priority': 0.8
                })
            elif progress < 0.7:
                recommendations.append({
                    'principle': principle.name,
                    'reason': 'Needs more practice',
                    'priority': 0.6
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        return recommendations[:3]

# ============================================================================
# STORY GENERATION ENGINE
# ============================================================================

class StoryGenre(Enum):
    """Story genres."""
    FANTASY = 1
    SCI_FI = 2
    ADVENTURE = 3
    MYSTERY = 4
    COMEDY = 5
    DRAMA = 6
    HORROR = 7
    SLICE_OF_LIFE = 8

@dataclass
class StoryElement:
    """Element of a generated story."""
    capsule_id: str
    role: str  # protagonist, antagonist, setting, prop, etc.
    description: str
    importance: float  # 0.0 to 1.0

@dataclass
class GeneratedStory:
    """A complete generated story."""
    id: str
    title: str
    genre: StoryGenre
    logline: str  # One-sentence summary
    premise: str  # Paragraph premise
    characters: List[StoryElement]
    settings: List[StoryElement]
    props: List[StoryElement]
    plot_points: List[str]
    themes: List[str]
    mood: str
    generated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'genre': self.genre.name,
            'logline': self.logline,
            'premise': self.premise,
            'characters': [{'capsule': c.capsule_id, 'role': c.role, 
                           'description': c.description} 
                          for c in self.characters],
            'settings': [{'capsule': s.capsule_id, 'role': s.role,
                         'description': s.description} 
                        for s in self.settings],
            'props': [{'capsule': p.capsule_id, 'role': p.role,
                      'description': p.description} 
                     for p in self.props],
            'plot_points': self.plot_points,
            'themes': self.themes,
            'mood': self.mood,
            'generated_at': self.generated_at
        }

class StoryGenerator:
    def _determine_mood(self, categorized: dict) -> str:
        """Determine the mood of the story based on characters or settings."""
        # Simple heuristic: if protagonist is brave, mood is 'heroic', else 'neutral'
        protagonist = next((c for c in categorized.get('characters', []) if getattr(c, 'role', '') == 'protagonist'), None)
        if protagonist and hasattr(protagonist, 'description') and 'brave' in protagonist.description.lower():
            return 'heroic'
        return 'neutral'

    def __init__(self, capsule_manager, memory_engine=None):
        self.capsule_manager = capsule_manager
        self.memory_engine = memory_engine
        self.generated_stories: List[GeneratedStory] = []

        # Story templates and tropes
        self.story_templates = self._initialize_templates()
        self.character_archetypes = self._initialize_archetypes()
        self.plot_twists = self._initialize_plot_twists()

    def _initialize_templates(self) -> Dict[StoryGenre, List[str]]:
        """Initialize story structure templates."""
        return {
            StoryGenre.FANTASY: [
                "The {character} must retrieve the {prop} from the {setting} to save the {setting2}",
                "A {character} discovers they have magical abilities related to {prop} in the {setting}",
                "The {character} and {character2} embark on a quest through {setting} to defeat {character3}"
            ],
            StoryGenre.SCI_FI: [
                "The {character} uncovers a conspiracy involving {prop} on {setting}",
                "A {character} must repair the {prop} before {setting} is destroyed",
                "The {character} and {character2} discover {prop} that changes their understanding of {setting}"
            ],
            StoryGenre.ADVENTURE: [
                "The {character} travels to {setting} in search of {prop}",
                "A {character} must navigate through {setting} while protecting {prop}",
                "The {character} and {character2} race against time in {setting} to find {prop}"
            ],
            StoryGenre.MYSTERY: [
                "The {character} investigates the disappearance of {prop} from {setting}",
                "A {character} must solve the mystery of {prop} before {setting} is affected",
                "The {character} uncovers secrets about {prop} that connect to {setting}"
            ],
            StoryGenre.COMEDY: [
                "The {character} accidentally acquires {prop} and chaos ensues in {setting}",
                "A {character} tries to use {prop} to impress others at {setting}",
                "The {character} and {character2} compete for {prop} in the most ridiculous ways at {setting}"
            ]
        }

    def _initialize_archetypes(self) -> Dict[str, List[str]]:
        """Initialize character archetypes."""
        return {
            'hero': ['brave warrior', 'clever inventor', 'wise mentor', 
                    'reluctant hero', 'chosen one'],
            'companion': ['loyal friend', 'comic relief', 'voice of reason',
                         'skillful sidekick', 'emotional support'],
            'antagonist': ['power-hungry villain', 'misunderstood rival',
                          'ancient evil', 'corrupt official', 'jealous peer'],
            'mentor': ['wise old guide', 'retired expert', 'mysterious stranger',
                      'parental figure', 'unlikely teacher'],
            'trickster': ['mischievous spirit', 'clever thief', 'shape-shifter',
                         'prankster', 'agent of chaos']
        }
    
    def _initialize_plot_twists(self) -> List[str]:
        """Initialize plot twist ideas."""
        return [
            "{character} was {character2} all along",
            "{prop} is actually sentient",
            "{setting} is an illusion",
            "The real villain was {character3}",
            "{prop} has the opposite effect",
            "{character} must become what they feared",
            "The goal was a distraction from the real problem"
        ]
    
    def generate_story(self, capsule_ids: List[str] = None, 
                      genre: StoryGenre = None) -> GeneratedStory:
        """
        Generate a story connecting capsules.
        If no capsules provided, uses interesting capsules from memory.
        """
        # Select capsules
        if capsule_ids:
            capsules = []
            for cap_id in capsule_ids:
                cap = self.capsule_manager.get_capsule_by_uuid(cap_id)
                if cap:
                    capsules.append(cap)
        else:
            capsules = self._select_interesting_capsules()
        
        if len(capsules) < 2:
            raise ValueError("Need at least 2 capsules to generate a story")
        
        # Select genre if not specified
        if not genre:
            genre = self._infer_genre(capsules)
        
        # Categorize capsules
        categorized = self._categorize_capsules(capsules)
        
        # Generate story elements
        title = self._generate_title(categorized, genre)
        logline = self._generate_logline(categorized, genre)
        premise = self._generate_premise(categorized, genre)
        plot_points = self._generate_plot_points(categorized, genre)
        themes = self._extract_themes(categorized)
        mood = self._determine_mood(categorized)
        
        # Create story object
        story = GeneratedStory(
            id=str(uuid.uuid4()),
            title=title,
            genre=genre,
            logline=logline,
            premise=premise,
            characters=categorized['characters'],
            settings=categorized['settings'],
            props=categorized['props'],
            plot_points=plot_points,
            themes=themes,
            mood=mood
        )
        
        self.generated_stories.append(story)
        return story
    
    def _select_interesting_capsules(self, count: int = 5) -> List:
        """Select interesting capsules for story generation."""
        capsules = self.capsule_manager.capsules
        
        if len(capsules) <= count:
            return capsules
        
        # Score capsules for story potential
        scored = []
        for capsule in capsules:
            score = self._story_potential_score(capsule)
            scored.append((score, capsule))
        
        # Sort by score and take top
        scored.sort(key=lambda x: x[0], reverse=True)
        return [capsule for _, capsule in scored[:count]]
    
    def _story_potential_score(self, capsule) -> float:
        """Score a capsule for story generation potential."""
        score = 0.0
        
        # Character capsules are great for stories
        if capsule.type.lower() == "character":
            score += 0.8
        
        # Pose capsules can suggest action
        if capsule.type.lower() == "pose":
            score += 0.6
        
        # Interesting names score higher
        name = capsule.name.lower()
        
        # Check for evocative words
        evocative_words = ['dragon', 'knight', 'wizard', 'robot', 'alien',
                          'castle', 'forest', 'mountain', 'ocean', 'city',
                          'sword', 'staff', 'crystal', 'book', 'key']
        
        for word in evocative_words:
            if word in name:
                score += 0.3
        
        # Metadata with description
        if isinstance(capsule.metadata, dict):
            if capsule.metadata.get('description'):
                score += 0.2
            
            # Check for story-relevant metadata
            story_metadata = ['backstory', 'personality', 'habitat', 'purpose']
            for key in story_metadata:
                if key in capsule.metadata:
                    score += 0.1
        
        # Usage frequency (popular capsules might be interesting)
        score += min(0.3, capsule.usage_count * 0.01)
        
        return min(1.0, score)
    
    def _infer_genre(self, capsules: List) -> StoryGenre:
        """Infer story genre from capsules."""
        # Analyze capsule names and metadata for genre clues
        genre_scores = defaultdict(float)
        
        genre_keywords = {
            StoryGenre.FANTASY: ['dragon', 'wizard', 'magic', 'castle', 'sword',
                               'elf', 'dwarf', 'spell', 'kingdom', 'quest'],
            StoryGenre.SCI_FI: ['robot', 'alien', 'spaceship', 'laser', 'cyborg',
                              'future', 'planet', 'tech', 'android', 'quantum'],
            StoryGenre.ADVENTURE: ['explorer', 'treasure', 'map', 'island', 'jungle',
                                 'mountain', 'ocean', 'pirate', 'artifact', 'journey'],
            StoryGenre.MYSTERY: ['detective', 'secret', 'clue', 'hidden', 'puzzle',
                               'code', 'evidence', 'suspect', 'crime', 'mystery'],
            StoryGenre.COMEDY: ['funny', 'silly', 'prank', 'joke', 'comic',
                              'absurd', 'ridiculous', 'wacky', 'humor', 'laugh']
        }
        
        for capsule in capsules:
            name = capsule.name.lower()
            metadata_str = str(capsule.metadata).lower()
            
            for genre, keywords in genre_keywords.items():
                for keyword in keywords:
                    if keyword in name or keyword in metadata_str:
                        genre_scores[genre] += 0.2
        
        # Default to adventure if no clear genre
        if not genre_scores:
            return StoryGenre.ADVENTURE
        
        # Return highest scoring genre
        return max(genre_scores.items(), key=lambda x: x[1])[0]
    
    def _categorize_capsules(self, capsules: List) -> Dict[str, List[StoryElement]]:
        """Categorize capsules into story roles."""
        categorized = {
            'characters': [],
            'settings': [],
            'props': []
        }
        
        for capsule in capsules:
            # Determine role based on type and name
            role = self._determine_story_role(capsule)
            description = self._generate_element_description(capsule, role)
            importance = self._determine_importance(capsule, role)
            
            element = StoryElement(
                capsule_id=capsule.uuid,
                role=role,
                description=description,
                importance=importance
            )
            
            if role in ['protagonist', 'companion', 'antagonist', 'mentor', 'trickster']:
                categorized['characters'].append(element)
            elif role in ['setting', 'location']:
                categorized['settings'].append(element)
            elif role in ['prop', 'artifact', 'weapon', 'tool']:
                categorized['props'].append(element)
        
        # Ensure at least one protagonist
        if not any(e.role == 'protagonist' for e in categorized['characters']):
            if categorized['characters']:
                # Make first character the protagonist
                categorized['characters'][0].role = 'protagonist'
        
        return categorized
    
    def _determine_story_role(self, capsule) -> str:
        """Determine a capsule's role in a story."""
        name = capsule.name.lower()
        capsule_type = capsule.type.lower()
        
        # Character roles
        if capsule_type == "character":
            # Check name for role hints
            if any(word in name for word in ['hero', 'protagonist', 'main']):
                return 'protagonist'
            elif any(word in name for word in ['villain', 'antagonist', 'evil']):
                return 'antagonist'
            elif any(word in name for word in ['friend', 'companion', 'sidekick']):
                return 'companion'
            elif any(word in name for word in ['mentor', 'teacher', 'guide']):
                return 'mentor'
            elif any(word in name for word in ['trickster', 'jester', 'fool']):
                return 'trickster'
            else:
                # Default based on usage
                if capsule.usage_count > 10:
                    return 'protagonist'
                else:
                    return 'companion'
        
        # Setting/location
        elif capsule_type in ["setting", "location"]:
            return 'setting'
        elif any(word in name for word in ['forest', 'castle', 'city', 'planet', 'room']):
            return 'setting'
        
        # Props/items
        elif capsule_type in ["prop", "item", "weapon"]:
            return 'prop'
        elif any(word in name for word in ['sword', 'staff', 'key', 'book', 'crystal']):
            return 'prop'
        
        # Pose capsules might be actions or character states
        elif capsule_type == "pose":
            if any(word in name for word in ['attack', 'defend', 'fight']):
                return 'action'
            else:
                return 'state'
        
        # Default
        return 'prop'
    
    def _generate_element_description(self, capsule, role: str) -> str:
        """Generate description for story element."""
        descriptions = []
        
        # Start with capsule name
        descriptions.append(capsule.name)
        
        # Add type information
        if capsule.type != role:
            descriptions.append(f"a {capsule.type}")
        
        # Add metadata description if available
        if isinstance(capsule.metadata, dict):
            desc = capsule.metadata.get('description')
            if desc:
                descriptions.append(f"described as: {desc}")
            
            # Add specific metadata based on role
            if role in ['protagonist', 'companion', 'antagonist']:
                personality = capsule.metadata.get('personality')
                if personality:
                    descriptions.append(f"personality: {personality}")
        
        # For settings, add ambiance
        if role == 'setting':
            ambiance_words = ['mysterious', 'ancient', 'futuristic', 'peaceful',
                            'dangerous', 'beautiful', 'haunted', 'high-tech']
            selected = random.choice(ambiance_words)
            descriptions.append(f"has a {selected} ambiance")
        
        return ". ".join(descriptions)
    
    def _determine_importance(self, capsule, role: str) -> float:
        """Determine importance of element in story."""
        importance = 0.5  # Base
        
        # Role-based importance
        role_importance = {
            'protagonist': 1.0,
            'antagonist': 0.9,
            'setting': 0.7,
            'prop': 0.6,
            'companion': 0.6,
            'mentor': 0.5,
            'trickster': 0.4
        }
        
        importance = role_importance.get(role, 0.5)
        
        # Adjust based on usage
        importance += min(0.3, capsule.usage_count * 0.02)
        
        return min(1.0, importance)
    
    def _generate_title(self, categorized: Dict, genre: StoryGenre) -> str:
        """Generate story title."""
        # Get main character
        protagonist = next((c for c in categorized['characters'] 
                          if c.role == 'protagonist'), None)
        
        # Get main setting
        main_setting = categorized['settings'][0] if categorized['settings'] else None
        
        # Get main prop
        main_prop = categorized['props'][0] if categorized['props'] else None
        
        # Genre-specific title patterns
        title_patterns = {
            StoryGenre.FANTASY: [
                "The {character} and the {prop}",
                "{character}'s {setting} Adventure",
                "Legend of the {prop}",
                "{character} of {setting}"
            ],
            StoryGenre.SCI_FI: [
                "{character}: {prop} Protocol",
                "The {setting} Incident",
                "{character} and the {prop}",
                "Digital {setting}"
            ],
            StoryGenre.ADVENTURE: [
                "{character}'s {setting} Quest",
                "The {prop} of {setting}",
                "{character} and the Lost {prop}",
                "Journey to {setting}"
            ]
        }
        
        patterns = title_patterns.get(genre, title_patterns[StoryGenre.ADVENTURE])
        pattern = random.choice(patterns)
        
        # Replace placeholders
        title = pattern
        if protagonist:
            cap = self.capsule_manager.get_capsule_by_uuid(protagonist.capsule_id)
            title = title.replace('{character}', cap.name if cap else "Hero")
        
        if main_setting:
            cap = self.capsule_manager.get_capsule_by_uuid(main_setting.capsule_id)
            title = title.replace('{setting}', cap.name if cap else "Unknown")
        
        if main_prop:
            cap = self.capsule_manager.get_capsule_by_uuid(main_prop.capsule_id)
            title = title.replace('{prop}', cap.name if cap else "Artifact")
        
        return title
    
    def _generate_logline(self, categorized: Dict, genre: StoryGenre) -> str:
        """Generate one-sentence logline."""
        # Get story elements
        protagonist = next((c for c in categorized['characters'] 
                          if c.role == 'protagonist'), None)
        antagonist = next((c for c in categorized['characters'] 
                         if c.role == 'antagonist'), None)
        main_setting = categorized['settings'][0] if categorized['settings'] else None
        main_prop = categorized['props'][0] if categorized['props'] else None
        
        # Get capsule names
        prot_name = "Someone"
        if protagonist:
            cap = self.capsule_manager.get_capsule_by_uuid(protagonist.capsule_id)
            prot_name = cap.name if cap else "A hero"
        
        antag_name = "a threat"
        if antagonist:
            cap = self.capsule_manager.get_capsule_by_uuid(antagonist.capsule_id)
            antag_name = cap.name if cap else "a villain"
        
        setting_name = "a mysterious place"
        if main_setting:
            cap = self.capsule_manager.get_capsule_by_uuid(main_setting.capsule_id)
            setting_name = cap.name if cap else "a dangerous location"
        
        prop_name = "a powerful artifact"
        if main_prop:
            cap = self.capsule_manager.get_capsule_by_uuid(main_prop.capsule_id)
            prop_name = f"the {cap.name}" if cap else "a mysterious object"
        
        # Genre-specific loglines
        loglines = {
            StoryGenre.FANTASY: [
                f"{prot_name} must recover {prop_name} from {setting_name} to defeat {antag_name}",
                f"A {prot_name.lower()} discovers {prop_name} holds the key to saving {setting_name} from {antag_name}",
                f"{prot_name} and their companions journey through {setting_name} seeking {prop_name} to stop {antag_name}"
            ],
            StoryGenre.ADVENTURE: [
                f"{prot_name} embarks on a quest through {setting_name} to find {prop_name} before {antag_name} does",
                f"To save {setting_name}, {prot_name} must locate {prop_name} and overcome {antag_name}",
                f"{prot_name}'s search for {prop_name} leads to {setting_name} and a confrontation with {antag_name}"
            ],
            StoryGenre.SCI_FI: [
                f"{prot_name} uncovers {prop_name} on {setting_name}, putting them in conflict with {antag_name}",
                f"A malfunction with {prop_name} forces {prot_name} to navigate the dangers of {setting_name} while evading {antag_name}",
                f"{prot_name} discovers that {prop_name} holds secrets about {setting_name} that {antag_name} will kill to possess"
            ]
        }
        
        # Default to adventure if genre not found
        genre_loglines = loglines.get(genre, loglines[StoryGenre.ADVENTURE])
        return random.choice(genre_loglines)
    
    def _generate_premise(self, categorized: Dict, genre: StoryGenre) -> str:
        """Generate paragraph-length premise."""
        logline = self._generate_logline(categorized, genre)
        
        # Expand logline into premise
        protagonist = next((c for c in categorized['characters'] 
                          if c.role == 'protagonist'), None)
        
        prot_desc = "Our hero"
        if protagonist:
            cap = self.capsule_manager.get_capsule_by_uuid(protagonist.capsule_id)
            if cap:
                prot_desc = cap.name
        
        # Add details based on genre
        genre_details = {
            StoryGenre.FANTASY: "In a world of magic and mystery,",
            StoryGenre.SCI_FI: "In a future of advanced technology,",
            StoryGenre.ADVENTURE: "In an epic journey of discovery,",
            StoryGenre.MYSTERY: "In a web of secrets and deception,",
            StoryGenre.COMEDY: "In a series of hilarious misadventures,"
        }
        
        start = genre_details.get(genre, "In an exciting adventure,")
        
        premise = f"{start} {logline} Along the way, they'll face challenges that test their courage and reveal hidden truths about themselves and the world around them."
        
        return premise
    
    def _generate_plot_points(self, categorized: Dict, genre: StoryGenre) -> List[str]:
        """Generate major plot points."""
        # Standard three-act structure
        plot_points = [
            "Setup: Introduce characters and world",
            "Inciting Incident: The event that starts the adventure",
            "First Act Break: Commitment to the journey",
            "Midpoint: Major revelation or twist",
            "Low Point: All seems lost",
            "Climax: Final confrontation",
            "Resolution: Aftermath and new normal"
        ]

        # Add specific details based on capsules
        protagonist = next((c for c in categorized['characters'] 
                          if c.role == 'protagonist'), None)
        antagonist = next((c for c in categorized['characters'] 
                         if c.role == 'antagonist'), None)
        main_prop = categorized['props'][0] if categorized['props'] else None

        if protagonist and main_prop:
            prot_cap = self.capsule_manager.get_capsule_by_uuid(protagonist.capsule_id)
            prop_cap = self.capsule_manager.get_capsule_by_uuid(main_prop.capsule_id)
            if prot_cap and prop_cap:
                plot_points[1] = f"{prot_cap.name} discovers {prop_cap.name}"

        if antagonist and protagonist:
            ant_cap = self.capsule_manager.get_capsule_by_uuid(antagonist.capsule_id)
            prot_cap = self.capsule_manager.get_capsule_by_uuid(protagonist.capsule_id)
            if ant_cap and prot_cap:
                plot_points[4] = f"{ant_cap.name} seems to defeat {prot_cap.name}"
                plot_points[5] = f"Final showdown between {prot_cap.name} and {ant_cap.name}"

        return plot_points
    
    def _extract_themes(self, categorized: Dict) -> List[str]:
        """Extract themes from story elements."""
        themes = []
        
        # Common themes
        common_themes = [
            "Friendship and loyalty",
            "Courage in the face of fear",
            "Self-discovery and growth",
            "The price of power",
            "Tradition vs. innovation",
            "Nature vs. technology",
            "Identity and belonging",
            "Sacrifice for the greater good"
        ]
        
        # Select 2-3 themes
        num_themes = random.randint(2, 3)
        selected = random.sample(common_themes, num_themes)
        themes.extend(selected)
        
        # Add theme based on protagonist if available
        protagonist = next((c for c in categorized['characters'] 
                          if c.role == 'protagonist'), None)
        
        if protagonist:
            cap = self.capsule_manager.get_capsule_by_uuid(protagonist.capsule_id)
            if cap and isinstance(cap.metadata, dict):
                personality = cap.metadata.get('personality', '').lower()
                
                if any(word in personality for word in ['brave', 'courageous']):
                    themes.append("Heroism and bravery")
                elif any(word in personality for word in ['clever', 'smart', 'wise']):
                    themes.append("Wisdom and intelligence over brute force")
                # elif any ... (incomplete condition removed)