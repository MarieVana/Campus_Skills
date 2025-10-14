#### Implementation of K-Means clustering algorithm from scratch
import numpy as np
import matplotlib.pyplot as plt
 
 # =========================
# 2) Classe KMeans
# =========================

class KMeans:
    """
    K-Means from scratch (NumPy).
    - n_init: on garde le run à WCSS minimale
    - random_state: pour reproductibilité
    - k : nombre de clusters
    - max_iter : nombre max d'itérations par run
    """
    ### Initialisation : “On choisit aléatoirement k points du dataset comme centroïdes de départ (méthode Forgy).”
    # Critère d’arrêt : “On répète assignation → mise à jour jusqu’à ce que les labels ne changent plus 
    # ou que max_iter soit atteint.
    
    
    def __init__(self, k, max_iter=300, n_init=1, random_state=None):
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = random_state
        # Attributs appris après fit
        self.centroids_ = None   # (k, d)
        self.labels_ = None      # (n,)
        self.inertia_ = None     # WCSS
        self.n_iter_ = None

    
    # --------- API publique ---------

    def prepare_data_for_kmeans(self, X, nan_strategy="impute", standardize=True):
        """Prépare X (float2D, k ≤ n, NaN, standardisation)."""
        prep = {}
        X = self._ensure_float_array(X)
        self._check_k_le_n(X)

        per_col, total = self._nan_report(X)
        prep['nan_per_col'], prep['nan_total'] = per_col, int(total)

        if total > 0:
            if nan_strategy == "impute":
                X = self._impute_nan_mean(X)
                prep['nan_strategy'] = 'impute_mean'
            elif nan_strategy == "drop":
                X, mask = self._drop_rows_with_nan(X)
                prep['nan_strategy'] = 'drop_rows'
                prep['row_mask_kept'] = mask
                self._check_k_le_n(X)
            else:
                raise ValueError("nan_strategy doit être 'impute' ou 'drop'")
        else:
            prep['nan_strategy'] = 'none'

        if standardize:
            mean, std = self._standardize_fit(X)
            X = self._standardize_transform(X, mean, std)
            prep['standardize'] = True
            prep['mean'], prep['std'] = mean, std
        else:
            prep['standardize'] = False

        return X, prep

   
    def fit(self, X):
        """Entraîne le modèle sur X. Retient le meilleur run (inertia minimale).
        fit : initialisation , distance, attribution des clusters, changement de centroid et ainsi de suite
        fit va appeler single_run n_init fois et retenir le meilleur"""
        
        X = self._ensure_float_array(X)
        best = None
        # Sous-seeds reproductibles avec SeedSequence
        ss = np.random.SeedSequence(self.random_state) ## seed principale
        children = ss.spawn(self.n_init)

        for run in range(self.n_init):
            rng = np.random.default_rng(children[run])
            out = self._single_run(X, rng)
            if (best is None) or (out["inertia_WCSS"] < best["inertia_WCSS"]):
                best = out
                
        self.centroids_ = best["centroids"]
        self.labels_ = best["labels"]
        self.inertia_ = best["inertia_WCSS"]
        self.n_iter_ = best["n_iter"]
        return self


    def predict(self, X):
        """Assigne les caracteristiques du modèle retenue aux nouveaux points en utilisant les centroïdes appris."""
        if self.centroids_ is None:
            raise RuntimeError("Appeler fit(X) avant predict.")
        X = self._ensure_float_array(X)
        D = self._distances(X, self.centroids_)
        return self._assign_labels_from_distances(D)

    def fit_predict(self, X):
        """renvoie juste les labels sur X après avoir réaliser le fit."""
        self.fit(X)
        return self.labels_

    def plot_clusters(self, X, title="K-Means clustering"):
        """Trace X (2D) coloré par cluster + centroïdes."""
        X = np.asarray(X, dtype=float)
        if X.shape[1] < 2:
            raise ValueError("plot_clusters attend ≥2 dimensions.")
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("Appeler fit(X) avant plot_clusters().")
        k = int(self.labels_.max()) + 1
        plt.figure()
        for j in range(k):
            mask = (self.labels_ == j)
            plt.scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.7, label=f"cluster {j}")
        plt.scatter(self.centroids_[:, 0], self.centroids_[:, 1],
                    s=120, marker='X', edgecolors='k', linewidths=1.0, label="centroids")
        plt.title(title); plt.xlabel("feat 1"); plt.ylabel("feat 2")
        plt.legend(); plt.grid(True); plt.show()

# ---------- méthodes privées (“coeur de KMeans”) ----------

######################################################
    ### 1/ Preparation of the dataset with private functions

    ## 1.a / Ensure X is a 2D float array to cast to float64 if necessary
    def _ensure_float_array(self,X):
        """Cast en float64, copie si nécessaire, vérifie la 2D."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X doit être 2D (n échantillons × d features).")
        return X

    ## 1.b / check that the number of clusters k <= n (number of samples)
    def _check_k_le_n(self,X):
        n = X.shape[0]
        if self.k > n:
            raise ValueError(f"k={self.k} > n={n}. Choisis k ≤ n.")

    ## 1.c/ Praparation of the dataset and handling of NaN values
    ### We will introduce some NaN values in the dataset to illustrate how to handle them
    def _nan_report(self,X):
        """Retourne nb de NaN par colonne et total."""
        nan_mask = np.isnan(X)
        per_col = nan_mask.sum(axis=0)
        total = nan_mask.sum()
        return per_col, total

    def _impute_nan_mean(self,X):
        """
        Impute NaN par la moyenne colonne par colonne (inplace sur une copie).
        Si une colonne entière est NaN, on la met à 0 
        """
        X = X.copy()
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_means, idx[1])
        return X

    def _drop_rows_with_nan(self,X):
        """Supprime les lignes contenant au moins un NaN."""
        mask = ~np.any(np.isnan(X), axis=1)
        return X[mask], mask


    ## 1.d) Standardization of the dataset
    def _standardize_fit(self,X, eps=1e-12):
        """
        Calcule mean et std par colonne. Clamp std pour éviter division par zéro.
        Retourne (mean, std_clamped).
        """
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std_clamped = np.where(std < eps, 1.0, std)
        return mean, std_clamped

    def _standardize_transform(self,X, mean, std):
        return (X - mean) / std

    ######################################################
    ##### full preparation function for K-Means dataset before clustering


# on va creation d'une partition de l'ensemble X en k clusters 
# - 1 /initier aléatoirement des centroids => ici la fonction make_blobs a initié directement les centroids dans le jeu de données
# on va donc les réucpérer. Mais 
# - 2 / répéter
#         - a) pour chaque observation Xi, calucler la distance euclidiennes des points avec les centroids
#         - b) associer chaque observation Xi au cluster dont le centre est le plus proche de Xi
#         - c) Recalculer les centroids
# - 3/ Reprendre au point 2/ jusquà ce que les centroids deviennent stables 

    def _initialize_centroids_random(self,X, k, seed=None):
        """
        X : data points avec n points en d dimensions
        k : nb de clusters
        """
        n = X.shape[0]
        if self.k > n:
            raise ValueError(f"k={self.k} > n={n}")
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=self.k, replace=False)  # indices distincts
        C = X[idx].copy()  # (k, d)
        return C


### compute pairwise squared distances between each point and each centroid 
## aide avec la formule : ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    def _distances(self,X, C):
        """
        Version mémoire-friendly: D = ||X||^2 + ||C||^2 - 2 X C^T
        D : (n, k) distances au carré entre chaque point et chaque centroide
        X : data points avec n points en d dimensions
        C : centroids avec k centroids en d dimensions
        """
        # ||X||^2 (n, )
        X_norm2 = np.sum(X * X, axis=1, keepdims=True)      # (n, 1)
        # ||C||^2 (k, )
        C_norm2 = np.sum(C * C, axis=1, keepdims=True).T    # (1, k)
        # produit (n, k)
        XC = X @ C.T
        D = X_norm2 + C_norm2 - 2.0 * XC
        # des petites erreurs numériques peuvent rendre D légèrement négatif
        np.maximum(D, 0.0, out=D)
        return D


### assign labels based on distance matrix D
### labels = numéro du cluster le plus proche, va de 0 à k-1
    def _assign_labels_from_distances(self,D):
        # labels[i] = argmin_j D[i, j]
        labels = np.argmin(D, axis=1) # attention labels de 0 à k-1
        return labels


### on va calculer l'inertie pour pouvoir évaluer la qualité du clustering
## ici ov prendre inertie comme la somme des distances au carré entre chaque point et le centroide de son cluster
## SSE = somme des distances au carré

## on va calculer l'inertie intra-cluster => point bien placés dans leur cluster 
    def _cluster_summaries(self,X, labels, k):
        """
        Retourne:
        counts: (k,) nb de points par cluster
        centroids: (k,d) moyennes par cluster
        wcss_per_cluster: (k,) inertie intra par cluster
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        counts = np.bincount(labels, minlength=k).astype(float)

        # Centroids via somme par cluster
        centroids = np.zeros((k, d), dtype=float)
        np.add.at(centroids, labels, X)
        nonempty = counts > 0
        centroids[nonempty] /= counts[nonempty, None]

        # WCSS par cluster
        diffs = X - centroids[labels]           # (n,d)
        sq = (diffs * diffs).sum(axis=1)        # (n,)
        wcss_per_cluster = np.zeros(k, dtype=float)
        np.add.at(wcss_per_cluster, labels, sq)

        return counts, centroids, wcss_per_cluster


### on calcul le TSS, BCSS, WCSS
## TSS = Total Sum of Squares = BSS + WCSS
## WCSS = Within-Cluster Sum of Squares = somme des distances au carré entre chaque point et le centroide de son cluster
## BSS = Between-Cluster Sum of Squares = somme des distances au carré entre chaque centroide et la moyenne globale pondérée par le nombre de points dans chaque cluster
    def _tss_bcss_wcss(self,X, labels, k):
        counts, C, wcss_k = self._cluster_summaries(X, labels, self.k)
        wcss = float(wcss_k.sum())

    # TSS (par rapport à la moyenne globale)
        xbar = X.mean(axis=0, dtype=float)
        tss = float(((X - xbar) ** 2).sum())

    # BCSS = TSS - WCSS  (équivalent à sum_j n_j ||mu_j - xbar||^2)
        bcss = tss - wcss
        return tss, bcss, wcss, counts, C, wcss_k

### update centroids as the mean of assigned points
    def _update_centroids_mean(self,X, labels, k):
        n, d = X.shape
        C = np.zeros((k, d), dtype=float)
        counts = np.bincount(labels, minlength=k).astype(float)
        np.add.at(C, labels, X)            # somme par cluster
        nonempty = counts > 0
        C[nonempty] /= counts[nonempty, None]
        return C, counts

    def _seed_from(self,random_state, run):
        """Combine un random_state (int) et l’index de run en graine 32 bits non signée."""
        if random_state is None:
            return None
        m32 = (1 << 32) - 1
        return hash((int(random_state), int(run))) & m32

# --- un run de KMeans ---
    # ---------- méthodes privées (ce que tu appelais “fonctions vues avant”) ----------
    def _single_run(self, X, rng):
        
        # 1) initialisation, centroids aléatoires
        C = self._initialize_centroids_random(X, rng)
        
        labels = None
        it = 0
        for it in range(1, self.max_iter + 1):
        # 2a) distances
            D = self._distances(X, C)
        # 2b) assignation
            new_labels = self._assign_labels_from_distances(D)
            if labels is not None and np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            
        # 2c) update
            C_new, counts = self._update_centroids_mean(X, labels, self.k)

            # clusters vides -> repositionner...
            empty = np.where(counts == 0)[0]
            if empty.size > 0:
                far_idx = np.argmax(np.min(D, axis=1))
                for j in empty: C_new[j] = X[far_idx]
            C = C_new
            
        # fin: calcule inertia intra_groupe.
        WCSS =  self._tss_bcss_wcss(X, labels, self.k)[2]
        TSS = self._tss_bcss_wcss(X, labels, self.k)[0]
        return {"centroids": C, "labels": labels, "inertia_WCSS": float(WCSS), "inertia_TSS": float(TSS), "n_iter": it}
    
    
    