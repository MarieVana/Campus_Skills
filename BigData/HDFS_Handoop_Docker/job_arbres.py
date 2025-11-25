from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("ArbresParis").getOrCreate()

df = (
    spark.read
    .option("header", "true")
    .option("sep", ";")
    .csv("file:///home/mvana/Module_BigData/Iteration2/arbresParis.csv")
)

# Pour vérifier les colonnes :
print("Colonnes du CSV :")
print(df.columns)
df.printSchema()

# Résultat intermédiaire
interm = (
    df.groupBy("Arrondissement21")
      .agg(F.count("*").alias("nb_arbres"))
)

interm.write.mode("overwrite").csv(
    "file:///home/mvana/Module_BigData/resultats/intermediaire_nb_arbres",
    header=True
)

# 4. Résultat final : top 5 arrondissements
final = interm.orderBy(F.col("nb_arbres").desc()).limit(5)

final.write.mode("overwrite").csv(
    "file:///home/mvana/Module_BigData/resultats/final_top5",
    header=True
)

spark.stop()
