# Real-World Applications

## 1. Medical Image Analysis
class MedicalImageAnalyzer:
    def __init__(self, manifold_method='umap', n_components=3):
        self.manifold_method = manifold_method
        self.n_components = n_components
        
    def analyze_medical_images(self, image_features, image_labels=None):
        """Analyze medical images with manifold learning"""
        
        # Manifold embedding
        if self.manifold_method == 'umap':
            reducer = SimpleUMAP(n_components=self.n_components, 
                               n_neighbors=15, min_dist=0.1)
        
        embedding = reducer.fit_transform(image_features)
        
        # Clustering to detect patterns
        clusterer = AdvancedManifoldClustering(
            manifold_method='precomputed',
            clustering_method='hdbscan',
            n_clusters=5
        )
        
        # Use existing embedding instead of re-embedding
        clusterer.X_embedded_ = embedding
        cluster_labels = clusterer.clusterer_.fit_predict(embedding)
        
        # Analyze results
        self._analyze_clusters(embedding, cluster_labels, image_labels)
        
        return embedding, cluster_labels
    
    def _analyze_clusters(self, embedding, cluster_labels, true_labels=None):
        """Analyze clustering quality"""
        
        unique_clusters = np.unique(cluster_labels)
        print(f"Number of clusters found: {len(unique_clusters)}")
        
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            
            print(f"Adjusted Rand Index: {ari:.3f}")
            print(f"Normalized Mutual Info: {nmi:.3f}")

## 2. Financial Market Analysis
class FinancialManifoldAnalyzer:
    def __init__(self):
        self.price_data = None
        self.returns_data = None
        self.manifold_embedding = None
        
    def load_stock_data(self, stock_prices):
        """Load stock price data"""
        self.price_data = stock_prices
        
        # Calculate returns
        self.returns_data = np.log(stock_prices / stock_prices.shift(1)).dropna()
        
        # Feature engineering
        self.features = self._create_financial_features()
        
    def _create_financial_features(self):
        """Create financial features"""
        
        features = {}
        
        # Technical indicators
        for window in [5, 10, 20]:
            features[f'sma_{window}'] = self.price_data.rolling(window).mean()
            features[f'volatility_{window}'] = self.returns_data.rolling(window).std()
        
        # Price ratios
        features['price_to_sma_20'] = self.price_data / features['sma_20']
        
        # Momentum features
        features['momentum_5'] = self.price_data / self.price_data.shift(5)
        features['momentum_20'] = self.price_data / self.price_data.shift(20)
        
        # Volume features (if available)
        # features['volume_sma'] = volume_data.rolling(20).mean()
        
        import pandas as pd
        feature_df = pd.DataFrame(features).dropna()
        
        return feature_df
    
    def perform_manifold_analysis(self):
        """Manifold analysis of financial data"""
        
        # Standardization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Manifold learning
        manifold_analyzer = ManifoldFeatureEngineer(
            manifold_methods=['pca', 'umap'],
            n_components=2
        )
        
        manifold_features = manifold_analyzer.fit_transform(features_scaled)
        
        # Regime detection
        regime_detector = AdvancedManifoldClustering(
            manifold_method='precomputed',
            clustering_method='hdbscan'
        )
        
        # Use manifold features
        regimes = regime_detector.fit_predict(manifold_features)
        
        self.manifold_features = manifold_features
        self.regimes = regimes
        self.scaler = scaler
        
        return manifold_features, regimes
    
    def predict_future_regime(self, new_prices):
        """Predict regime for new data"""
        
        # Calculate new features
        # (implementation similar to _create_financial_features)
        
        # Transform with existing scaler
        new_features_scaled = self.scaler.transform(new_features)
        
        # Manifold transform
        new_manifold_features = self.manifold_analyzer.transform(new_features_scaled)
        
        # Predict regime
        new_regimes = self.regime_detector.predict(new_manifold_features)
        
        return new_regimes

## 3. Social Network Analysis
class SocialNetworkManifoldAnalyzer:
    def __init__(self):
        self.graph = None
        self.node_features = None
        self.embeddings = None
        
    def analyze_social_network(self, adjacency_matrix, user_features=None):
        """Analyze social network with manifold learning"""
        
        # Graph embedding
        graph_features = self._extract_graph_features(adjacency_matrix)
        
        # Combine with user features
        if user_features is not None:
            combined_features = np.hstack([graph_features, user_features])
        else:
            combined_features = graph_features
        
        # Manifold analysis
        manifold_reducer = SimpleUMAP(n_components=2, n_neighbors=10)
        embeddings = manifold_reducer.fit_transform(combined_features)
        
        # Community detection
        community_detector = AdvancedManifoldClustering(
            manifold_method='precomputed',
            clustering_method='hdbscan'
        )
        
        communities = community_detector.fit_predict(embeddings)
        
        # Analyze communities
        self._analyze_communities(adjacency_matrix, communities, embeddings)
        
        return embeddings, communities
    
    def _extract_graph_features(self, adjacency_matrix):
        """Extract features from graph"""
        
        features = []
        
        # Basic features
        degrees = np.sum(adjacency_matrix, axis=1)  # Degree of each node
        features.append(degrees.reshape(-1, 1))
        
        # Clustering coefficient
        clustering_coeffs = self._compute_clustering_coefficient(adjacency_matrix)
        features.append(clustering_coeffs.reshape(-1, 1))
        
        # Centrality measures
        betweenness = self._compute_betweenness_centrality(adjacency_matrix)
        features.append(betweenness.reshape(-1, 1))
        
        # Local features
        local_features = self._compute_local_features(adjacency_matrix)
        features.append(local_features)
        
        return np.hstack(features)
    
    def _compute_clustering_coefficient(self, adj_matrix):
        """Compute clustering coefficient"""
        
        clustering_coeffs = []
        
        for i in range(len(adj_matrix)):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            
            if len(neighbors) < 2:
                clustering_coeffs.append(0)
                continue
            
            # Number of edges between neighbors
            edges_between_neighbors = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and adj_matrix[j, k] > 0:
                        edges_between_neighbors += 1
            
            # Maximum possible edges
            max_edges = len(neighbors) * (len(neighbors) - 1) / 2
            
            clustering_coeff = edges_between_neighbors / max_edges
            clustering_coeffs.append(clustering_coeff)
        
        return np.array(clustering_coeffs)
    
    def _compute_betweenness_centrality(self, adj_matrix):
        """Compute betweenness centrality (simplified version)"""
        n_nodes = len(adj_matrix)
        betweenness = np.zeros(n_nodes)
        
        # Simplified calculation - for full implementation use networkx
        for i in range(n_nodes):
            # Count shortest paths passing through node i
            # This is a simplified version
            neighbors = np.where(adj_matrix[i] > 0)[0]
            betweenness[i] = len(neighbors) / (n_nodes - 1)
        
        return betweenness
    
    def _compute_local_features(self, adj_matrix):
        """Compute local structural features"""
        
        local_feats = []
        
        for i in range(len(adj_matrix)):
            # Local density
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                local_density = np.mean(adj_matrix[neighbors][:, neighbors])
            else:
                local_density = 0
            
            # Triangle count
            triangles = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and adj_matrix[j, k] > 0:
                        triangles += 1
            
            local_feats.append([local_density, triangles])
        
        return np.array(local_feats)
    
    def _analyze_communities(self, adj_matrix, communities, embeddings):
        """Analyze detected communities"""
        
        unique_communities = np.unique(communities)
        print(f"Number of communities found: {len(unique_communities)}")
        
        # Community sizes
        for comm in unique_communities:
            if comm != -1:  # Exclude noise
                size = np.sum(communities == comm)
                print(f"Community {comm}: {size} nodes")
        
        # Modularity calculation (simplified)
        modularity = self._compute_modularity(adj_matrix, communities)
        print(f"Modularity: {modularity:.3f}")
    
    def _compute_modularity(self, adj_matrix, communities):
        """Compute modularity of community structure"""
        
        m = np.sum(adj_matrix) / 2  # Total number of edges
        if m == 0:
            return 0
        
        modularity = 0
        degrees = np.sum(adj_matrix, axis=1)
        
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if communities[i] == communities[j] and communities[i] != -1:
                    expected = degrees[i] * degrees[j] / (2 * m)
                    modularity += (adj_matrix[i, j] - expected)
        
        return modularity / (2 * m)

## 4. Time Series Analysis with Manifold Learning
class TimeSeriesManifoldAnalyzer:
    def __init__(self, window_size=10, manifold_method='umap'):
        self.window_size = window_size
        self.manifold_method = manifold_method
        
    def analyze_time_series(self, time_series):
        """Analyze time series using manifold learning"""
        
        # Create sliding windows
        windowed_data = self._create_windows(time_series)
        
        # Manifold embedding
        if self.manifold_method == 'umap':
            reducer = SimpleUMAP(n_components=2, n_neighbors=10)
        elif self.manifold_method == 'pca':
            reducer = AdvancedPCA(n_components=2)
        
        embedding = reducer.fit_transform(windowed_data)
        
        # Detect patterns/regimes
        clusterer = AdvancedManifoldClustering(
            manifold_method='precomputed',
            clustering_method='dbscan'
        )
        
        patterns = clusterer.fit_predict(embedding)
        
        return embedding, patterns, windowed_data
    
    def _create_windows(self, time_series):
        """Create sliding windows from time series"""
        
        windows = []
        for i in range(len(time_series) - self.window_size + 1):
            window = time_series[i:i + self.window_size]
            windows.append(window)
        
        return np.array(windows)
    
    def detect_anomalies_in_time_series(self, time_series):
        """Detect anomalies in time series using manifold learning"""
        
        # Create windows
        windowed_data = self._create_windows(time_series)
        
        # Use manifold anomaly detector
        anomaly_detector = ManifoldAnomalyDetector(
            manifold_method=self.manifold_method,
            detection_method='ensemble',
            contamination=0.05
        )
        
        # Fit on normal data (assume first 80% is normal)
        split_point = int(0.8 * len(windowed_data))
        anomaly_detector.fit(windowed_data[:split_point])
        
        # Detect anomalies in all data
        predictions, scores = anomaly_detector.predict(windowed_data)
