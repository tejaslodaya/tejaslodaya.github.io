---
layout: post
title: Building a spatial querying and data manipulation engine
---
*   [Introduction](#intro)
*   [Basic requirements](#requirements)
*   [Query support](#query)
*   [Example use-cases](#example)
*   [Why not x?](#whynot)
    *   [Why not PostgreSQL + PostGIS?](#post)
    *   [Why not Hadoop-GIS or Spatial Hadoop?](#hadoop)
    *   [Why not LocationSpark, GeoSpark (Apache Sedona), Magellan, SpatialSpark?](#spark)
    *   [Why GeoMesa?](#geomesa)
*   [Conclusion](#conclusion)
*   [References](#refer)

### Introduction {#intro}

Spatial computing and emergence of massive scale spatial data is becoming increasingly important with the proliferation of mobile devices, cost effective and ubiquitous positioning technologies, development of high resolution imaging technologies, and contribution from a large number of community users. With startups like Uber, Instacart, Grubhub generating, ingesting, storing and processing massive amounts of spatial data, leading the spatial data revolution. However, the needs of storing and processing large-scale spatial data are poorly met by current general-purpose storage systems and calls for a more efficient spatio-temporal data management system. 

In this blog-post, we’ll talk about the do’s and don’ts of location data management and explore possible ways of storing and retrieving OpenStreetMap (OSM) data into the big-data ecosystem.

### Basic requirements {#requirements}

1. **Ease of use, versatile** - easy querying and retrieving data as Spark dataframes, with robust APIs for Scala, Python and SparkSQL
2. **Cost effective** - avoid storage of data in RDS or in-memory due to exorbitant cost of keeping the instance up all the time. Disk-based storage (AWS S3) or cold storage (AWS Glacier) preferred
3. **Fault tolerant**  - continue to operate even in the presence of node failures. Avoid single-node RDS monoliths.
4. **Fast, interactive querying** - ability to run interactive analytic queries with compute being horizontally scalable, rather than vertical.

### Query support {#query}

1. **kNN queries**- takes a set of points R, a query point q, and an integer k ≥ 1 as input, and finds the k nearest points in R to q
2. **Range queries** - takes range R and a set of geometric objects S, and returns all objects in S that lie in the range R
3. **Spatial joins** - takes two input sets of spatial records R and S and a join predicate θ (e.g., *overlap, intersect, contains*, etc..)
4. **Distance joins** - special case of spatial join where the join predicate is *withindistance*

### Example use-cases {#example}

1. **Traffic analysis** - hourly speed profile of the entire city, with a formula of distance/time between successive pings, uses *st_distanceSpheroid*
2. **Map matching** - snaps driver’s noisy GPS coordinates to the underlying road network using stochastic state space models, uses *st_distance*
3. **Polygon coverage** - find number of trips originating from a polygon over total number of trips, uses *st_contains*
4. **Address distribution** - find the number of customer addresses within x meters of each other, uses *st_distanceSphere*

<br/>
Pictorically, 
<br/>
<br/>
<img src="/assets/img/geomesa.png" alt="geomesa" width="70%" height="50%"/>

<em> Figure 1: Representation of data warehouses and querying engines</em>

### Why not x? {#whynot}

Geospatial querying can be accomplished by PostgreSQL with PostGIS extension, Spatial Hadoop, LocationSpark and GeoMesa. 

#### Why not PostgreSQL + PostGIS? {#post}

1. PostgreSQL as a datastore is not horizontally scalable, only vertically scalable. This can cause potential bottlenecks when terabytes of data is stored and analysed.
2. The Spark-JDBC connector used to convert PostgreSQL results back to Spark dataframe implements query-pushdown. At large-scale, this design makes PostgreSQL I/O bound rather than compute-bound.
3. Fails to leverage the power of distributed memory and cost-effectiveness of big-data.
4. Susceptible to failures, since this is a single node, vertically scalable RDS monolith

Full performance comparison is published at VLDB 2013, as part of [Hadoop-GIS: A High Performance Spatial Data Warehousing System over MapReduce](http://www.vldb.org/pvldb/vol6/p1009-aji.pdf) benchmarks.

#### Why not Hadoop-GIS or Spatial Hadoop? {#hadoop}

Hadoop-GIS is unable to reuse intermediate data and writes intermediate results back to HDFS. (not just Hadoop-GIS, but Hadoop in general).

#### Why not LocationSpark, GeoSpark (Apache Sedona), Magellan, SpatialSpark? {#spark}

To address the challenges faced by Hadoop, in-memory cluster computing frameworks for processing large-scale spatial data were developed based on Spark.

1. **Apache Sedona** - still in incubation and active development, no kNN joins
2. **LocationSpark** - limited data types, no recent development
3. **Magellan** - high shuffle costs, no range query optimizations
4. **SpatialSpark** - high memory costs, no recent development

For a detailed analysis, please refer to [How Good Are Modern Spatial Analytics Systems? - VLDB ‘18](http://www.vldb.org/pvldb/vol11/p1661-pandey.pdf)

#### Why GeoMesa? {#geomesa}

GeoMesa provides a suite of tools to manage and analyze huge spatio-temporal datasets and is a full-fledged platform. 

1. Provides support for near real time stream processing of spatio-temporal data by layering on top of Apache Kafka
2. Supports a host of data-stores like Cassandra, Bigtable, Redis, FileSystem (S3, HDFS), Kafka and Accumulo. 
3. Supports Spark Analytics, with robust APIs for Scala-spark, SparkSQL and PySpark - can be used only for distributed querying and analytical needs.
4. Horizontally scalable (add more nodes to add more capacity)
5. Provides JSON, Parquet, Shapefile converters for ingesting data into GeoMesa
6. Mature and under active development, with community support (7 years old, [latest 3.1.0 release](https://github.com/locationtech/geomesa/releases/tag/geomesa_2.11-3.1.0) on 28th October, 2020)

### Conclusion {#conclusion}

GeoMesa seems a clear winner and checks all the requirements for a fully-stable, cost-effective, fault tolerant, large-scale geospatial analytics platform.

Some advice on how to set-up :

1. Choose AWS S3, Hadoop HDFS, Google FileStorage or Azure BlobStore as the datastore, depending on the rest of your environment
2. Partition the data into folders, by date, by city or by country, depending on your scale and use-case
3. Create a cron job to download OSM data from [here](http://download.geofabrik.de/) (to avoid staleness, say weekly)
4. Ingest and convert data obtained from #3 into GeoMesa format, index and store it
5. Use the [FileSystem RDD Provider](https://www.geomesa.org/documentation/stable/user/spark/providers.html#filesystem-rdd-provider) inside GeoMesa Spark and run any spark based spatial workloads!

### References {#refer}

1. [Code] [GeoMesa - NYC Taxis](https://databricks.com/notebooks/GeoMesa-NYC-Taxis.html), [GeoMesa + H3 Notebook](https://databricks.com/notebooks/geomesa-h3-notebook.html)
2. [Paper] [Hadoop-GIS: A High Performance Spatial Data Warehousing System over MapReduce](http://www.vldb.org/pvldb/vol6/p1009-aji.pdf)
3. [Book] [An Architecture for Fast and General Data Processing on Large Clusters - M.  Zaharia - ACM '16](https://dl.acm.org/doi/book/10.1145/2886107)
3. [Benchmark] [Benchmarking of Big Data Technologies for Ingesting and Querying Geospatial Datasets](https://www.reply.com/en/topics/big-data-and-analytics/Shared%20Documents/DSTL-Report-Data-Reply-2017.pdf)
4. [Benchmark]  [How Good Are Modern Spatial Analytics Systems?](http://www.vldb.org/pvldb/vol11/p1661-pandey.pdf)
5. [Github] [locationtech/geomesa](https://github.com/locationtech/geomesa), [apache/incubator-sedona](https://github.com/apache/incubator-sedona/)
