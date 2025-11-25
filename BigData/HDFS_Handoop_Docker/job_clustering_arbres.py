#!/home/mvana/bigdata/bin/python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans


# ------------------------------
# 1. Création de la SparkSession
# ------------------------------
spark = (
    SparkSession.builder
    .appName("Clustering_Arbres_Paris")
    # .master("spark://51.91.85.211:7077")  # à activer si tu veux forcer le master
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ------------------------------
# 2. Chargement du CSV
# ------------------------------

# Chemin LOCAL (comme tu le fais déjà)
csv_path = "file:///home/mvana/Module_BigData/Iteration2/arbresParis.csv"

print("Lecture du fichier :", csv_path)

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("sep", ";")   # adapte si besoin, mais pour Paris c'est souvent ';'
    .csv(csv_path)
)

print("Nb de lignes initial :", df.count())
print("Quelques colonnes dispo :", df.columns)

# ------------------------------
# 3. Sélection / renommage des colonnes utiles
# ------------------------------

# On essaie de trouver le bon nom de colonne pour l'arrondissement
if "arrondissement3" in df.columns:
    col_arr = "arrondissement3"
elif "Arrondissement21" in df.columns:
    col_arr = "Arrondissement21"
elif "arrondissement" in df.columns:
    col_arr = "arrondissement"
else:
    raise ValueError("Impossible de trouver la colonne d'arrondissement dans le CSV.")

df_sel = (
    df.select(
        F.trim(F.col(col_arr)).alias("arrondissement"),
        F.col("circonference en cm").cast("double").alias("circonference"),
        F.col("hauteur en m").cast("double").alias("hauteur"),
    )
)

print("Nb de lignes après select :", df_sel.count())

# ------------------------------
# 4. Nettoyage simple
#    - on enlève les lignes vides / nulles
#    - on enlève les valeurs <= 0 (pas très réalistes)
# ------------------------------

df_clean = (
    df_sel
    .filter(F.col("arrondissement") != "")
    .filter(F.col("circonference").isNotNull() & (F.col("circonference") > 0))
    .filter(F.col("hauteur").isNotNull() & (F.col("hauteur") > 0))
)

nb_propres = df_clean.count()
print("Nb lignes propres :", nb_propres)

if nb_propres == 0:
    print("Aucune ligne valide après nettoyage, j'arrête le clustering.")
    spark.stop()
    exit(0)

print("Exemple de lignes propres :")
df_clean.show(5, truncate=False)

# ------------------------------
# 5. Construction du vecteur de features
#    (circonference, hauteur)
# ------------------------------

assembler = VectorAssembler(
    inputCols=["circonference", "hauteur"],
    outputCol="features",
)

data = assembler.transform(df_clean)

# ------------------------------
# 6. K-Means
# ------------------------------

k = 4  # à adapter si tu veux tester d'autres valeurs
kmeans = KMeans(
    k=k,
    seed=42,
    featuresCol="features",
    predictionCol="cluster",
)

print(f"Lance le KMeans avec k={k} ...")
model = kmeans.fit(data)

# Centres des clusters
centers = model.clusterCenters()
print("Centres des clusters (circonference, hauteur) :")
for i, c in enumerate(centers):
    print(f"  Cluster {i} : {c}")

# Application du modèle
result = model.transform(data)

print("Quelques lignes avec le cluster assigné :")
result.select("arrondissement", "circonference", "hauteur", "cluster").show(10, truncate=False)

# ------------------------------
# 7. Sauvegarde des résultats en LOCAL
#    (tu pourras ensuite les pousser dans HDFS)
# ------------------------------

output_dir = "/home/mvana/Module_BigData/resultats_clustering_arbres"

(
    result
    .select("arrondissement", "circonference", "hauteur", "cluster")
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(output_dir)
)

print("Résultats sauvegardés dans :", output_dir)
print("Tu peux ensuite faire :")
print("  sudo docker cp /home/mvana/Module_BigData/resultats_clustering_arbres namenode:/tmp/resultats_clustering_arbres")
print("  sudo docker exec -it namenode bash")
print("  hdfs dfs -mkdir -p /data_test/resultats_clustering_arbres")
print("  hdfs dfs -put /tmp/resultats_clustering_arbres /data_test/resultats_clustering_arbres")

spark.stop()
