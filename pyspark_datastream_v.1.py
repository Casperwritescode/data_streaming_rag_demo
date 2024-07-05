from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, current_timestamp, udf, col
from pyspark.sql.types import StringType
import re
from pyspark.sql.functions import col
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("RAG-Processing").getOrCreate()

# Define paths
raw_docs_path = "dbfs:/test_docs"
processed_docs_path = "dbfs:/rag_demo/processed_docs"
raw_docs_checkpoint_path = "dbfs:/rag_demo/checkpoints/raw_documents"
processed_docs_checkpoint_path = "dbfs:/rag_demo/checkpoints/processed_docs"
parsed_docs_checkpoint_path = "dbfs:/rag_demo/checkpoints/parsed_documents"

# Define a simple parsing function
def parse_content(content):
    # Example parsing logic: removing non-alphanumeric characters
    parsed_content = re.sub(r'\W+', ' ', content)
    return parsed_content

# Register the parsing function as a UDF
parse_udf = udf(parse_content, StringType())

# Step 1: Read text files from DBFS and write to raw_documents table
raw_docs_stream = (
    spark.readStream
    .format("text")
    .option("path", raw_docs_path)
    .load()
    .withColumn("content", col("value"))
    .withColumn("id", input_file_name())
    .withColumn("ingest_time", current_timestamp())
    .drop("value")
)

raw_docs_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", raw_docs_checkpoint_path) \
    .option("mergeSchema", "true") \
    .table("raw_documents")

# Step 2: Read from raw_documents, process content, and write to processed_docs
raw_documents_stream = spark.readStream.table("raw_documents")

processed_docs_stream = raw_documents_stream.withColumn("content", col("content"))

processed_docs_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", processed_docs_checkpoint_path) \
    .option("mergeSchema", "true") \
    .table("processed_docs")

# Step 3: Read from processed_docs, parse content, and write to parsed_documents
processed_docs_schema = "id STRING, ingest_time TIMESTAMP, content STRING"

processed_documents_stream = (
    spark.readStream
    .format("delta")
    .option("checkpointLocation", processed_docs_checkpoint_path)
    .table("processed_docs")
)

parsed_documents_stream = processed_documents_stream.withColumn("parsed_content", parse_udf("content"))

parsed_documents_stream = parsed_documents_stream.withColumn("ingest_time", current_timestamp())

parsed_documents_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", parsed_docs_checkpoint_path) \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .option("mergeSchema", "true") \
    .table("parsed_documents")

# Step 4: Process parsed_documents for search_context (assuming further processing if needed)
parsed_documents_schema = "id STRING, ingest_time TIMESTAMP, parsed_content STRING"

parsed_documents_stream = (
    spark.readStream
    .format("delta")
    .option("checkpointLocation", parsed_docs_checkpoint_path)
    .table("parsed_documents")
)

search_context_stream = parsed_documents_stream.select("id", "ingest_time", "content")

search_context_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", "dbfs:/rag_demo/checkpoints/search_context") \
    .outputMode("append") \
    .table("search_context")
