from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, current_timestamp, udf, col
from pyspark.sql.types import StringType, StructType, StructField
import re
from transformers import pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("RAG-Processing").getOrCreate()

# Define paths
raw_docs_path = "dbfs:/test_docs"
processed_docs_table_path = "dbfs:/rag_demo/processed_docs"
parsed_docs_table_path = "dbfs:/rag_demo/parsed_documents"
search_context_table_path = "dbfs:/rag_demo/search_context"
raw_docs_checkpoint_path = "dbfs:/rag_demo/checkpoints/raw_documents"
processed_docs_checkpoint_path = "dbfs:/rag_demo/checkpoints/processed_docs"
parsed_docs_checkpoint_path = "dbfs:/rag_demo/checkpoints/parsed_documents"
search_context_checkpoint_path = "dbfs:/rag_demo/checkpoints/search_context"

# Define schema for raw documents
raw_schema = StructType([
    StructField("value", StringType(), True)
])

# Define a simple parsing function
def parse_content(content):
    parsed_content = re.sub(r'\W+', ' ', content)
    return parsed_content

# Register the parsing function as a UDF
parse_udf = udf(parse_content, StringType())

# Load the BART summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define a UDF for text summarization
def summarize_text(text):
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return text  # In case of any error, return the original text

# Register the summarization UDF
summarize_udf = udf(summarize_text, StringType())

# Step 1: Read text files from DBFS and write to raw_documents table
raw_docs_stream = (
    spark.readStream
    .format("text")
    .schema(raw_schema)  # Define schema explicitly
    .load(raw_docs_path)
    .withColumn("content", col("value"))
    .withColumn("id", input_file_name())
    .withColumn("ingest_time", current_timestamp())
    .drop("value")
)

raw_docs_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", raw_docs_checkpoint_path) \
    .outputMode("append") \
    .start("raw_documents")

# Step 2: Read from raw_documents, process content, and write to processed_docs
raw_documents_stream = spark.readStream.table("raw_documents")

processed_documents_stream = raw_documents_stream.withColumn("processed_content", col("content"))

processed_documents_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", processed_docs_checkpoint_path) \
    .outputMode("append") \
    .start("processed_docs")

# Step 3: Read from processed_docs, parse content, and write to parsed_documents
processed_documents_stream = spark.readStream.table("processed_docs")

parsed_documents_stream = processed_documents_stream.withColumn("processed_content", col("content"))

parsed_documents_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", parsed_docs_checkpoint_path) \
    .outputMode("append") \
    .start("parsed_documents")

# Step 4: Read from parsed_documents, summarize content, and write to search_context
parsed_documents_stream = spark.readStream.table("parsed_documents")

# Apply summarization
summarized_documents_stream = parsed_documents_stream.withColumn("processed_content", summarize_udf(col("content")))

# Select columns for search_context
search_context_stream = summarized_documents_stream.select("id", "ingest_time", col("processed_content").alias("content"))

search_context_stream.writeStream \
    .format("delta") \
    .option("checkpointLocation", search_context_checkpoint_path) \
    .outputMode("append") \
    .start(search_context_table_path)
