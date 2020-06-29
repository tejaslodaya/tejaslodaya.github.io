---
layout: post
title: Faster Spark joins
---
*   [Introduction](#introduction)
*   [Shuffle hash join/ reduce-side join](#shufflehashjoin)
*   [partitionBy](#partitionBy)
*   [Misc partitioning tips](#misc)
*   [Broadcast join / map-side join](#broadcast)
*   [Even faster map-side joins](#fasterjoins)
*   [Filter Large RDD](#filter)
*   [References](#references)

<h3 id="introduction"> Introduction </h3>
---
Some of the most useful operations we get with keyed data comes from using it together with other keyed data. Joining data together is probably one of the most common operations on a pair RDD, and we have a full range of options including right and left outer joins, cross joins, and inner joins. In order to join data, spark needs each key-to-be-joined to exist on the same partition.  There are several ways to achieve co-location which are documented below

<h3 id="shufflehashjoin"> Shuffle hash join/ reduce-side join </h3>
---
By default, when a RDD is read from textfile or HDFS, it follows the partitioning scheme of hadoop filesystem. As seen above, a pre-requisite for join is co-location.This operation will hash all the keys of both datasets, sending elements with the same key hash across the network to the same machine, and then join together the elements with the same key on that machine. 

For example, *userData* and *events* are shuffled causing heavy network traffic.
<img src="https://learning.oreilly.com/library/view/learning-spark/9781449359034/assets/lnsp_0404.png" width="70%" height="50%"/>

This operation is also called reduce-side join because the actual join process happens in the reduce phase. It follows the traditional map-shuffle-reduce flow.

<h3 id="partitionBy"> partitionBy </h3>
---
In cases where a largeRDD is used repeatedly, over and over again, shuffle-hash join would cause re-shuffling on each iteration making it costly. To avoid this, largeRDD is partitioned using RangePartitioner/HashPartitioner and spark uses this information to make the smallRDD use the same partitioner to find the partition to which the keys goes to. This way, a huge shuffle is avoided.

Note:

1. Use of *partitionBy* before any action is performed on largeRDD doesn't cost extra, since it is lazily evaluated.
2. Persist the RDD just after its been partitioned, if forgot would cause re-evaluation of RDD's complete lineage

<script src="https://gist.github.com/tejaslodaya/26b8c25cbf222efc6d9d51ac7d8bfb64.js"></script>

<img src="https://learning.oreilly.com/library/view/learning-spark/9781449359034/assets/lnsp_0405.png" width="70%" height="50%"/>

Functions other than *join* which take partitioning as advantage are *cogroup(), groupWith(), leftOuterJoin(), rightOuterJoin(), groupByKey(), reduceByKey(), combineByKey()*, and *lookup()*

<h3 id="misc"> Misc partitioning tips </h3>
---
ShuffleHashJoin can be avoided in the below scenarios

1. Both tables use the same partitioner
	<script src="https://gist.github.com/tejaslodaya/b02cf0b42593910a0b39a05dc761ab59.js"></script>
	
2. Second RDD is a derivative of First RDD - 

	Assume First RDD is hash partitioned and Second RDD is derived by using *mapValues* on First RDD. This way, both are cached on the same machine. 
3. If one of the RDDs is already shuffled before -
	
	Many spark operations automatically result in an RDD with known partitioning information and join takes advantage of this information. For example, sortByKey and groupByKey result in a partitioned RDD, with a valid non-default partitioner.  This behaves the same way as of `partitionBy` in [(2)](#partitionBy)
		
	![](https://web.archive.org/web/20190325153819if_/https://blog.cloudera.com/wp-content/uploads/2014/03/spark-devs1.png)
	
	Above, *B* is not shuffled when joined with *F* because *groupBy* is applied on *B*. 

<h3 id="broadcast"> Broadcast join / map-side join </h3>
---	
In order to avoid the shuffle-reduce phase, join operation is delegated to map-stage where-in, one of the tables (smaller one) is broadcasted in-memory to each mapper. This works only when one of the tables is relatively small.

Below are the steps to perform broadcast join:

1. Create a RDD for both tables on which join is to be performed
2. Download Small RDD to the driver, create *map* and broadcast on each execution node
3. Map over each row of Large RDD, retrieve value (from Small RDD) using key from the iterator of Large RDD.
4. Broadcast join will be executed concurrently for each partition since each partition has its own copy of the small RDD.

<script src="https://gist.github.com/tejaslodaya/c8219918b25b223f44dbf4d970af3463.js"></script>

<h3 id="fasterjoins"> Even faster map-side joins </h3>
---	
There are some scaling problems with map-side join. When thousands of mappers read the small join table from the Hadoop Distributed File System (HDFS) into memory at the same time, the join table easily becomes the performance bottleneck, causing the mappers to time out during the read operations. 

The basic idea of optimization is to create a new MapReduce local task just before the original join MapReduce task. This new task reads the small table data from HDFS to an in-memory hash table. After reading, it serializes the in-memory hash table into a hashtable file. In the next stage, when the MapReduce task is launching, it uploads this hashtable file to the Hadoop distributed cache, which populates these files to each mapper's local disk. So all the mappers can load this persistent hashtable file back into memory and do the join work as before. 

After optimization, the small table needs to be read just once. Also if multiple mappers are running on the same machine, the distributed cache only needs to push one copy of the hashtable file to this machine.

<h3 id="filter"> Filter Large RDD</h3>
---	
When joining an extremely large table and a subset of this table, a huge shuffle takes place. Join causes majority of the large table to drop. For example, when you're joining two RDDs namely *worldRDD* and *indiaRDD*, a join would cause majority of *worldRDD* to drop. An extremely fast (10x speedup) is to filter the *worldRDD* using the keys of *indiaRDD* and then performing a join.

This method is faster and causes less data to be shuffled over the network. 

Note:

1. Always explore the distribution of keys before performing a full-blown shufflejoin.
2. The efficiency gain here depends on the filter operation that reduces the size of larger RDD. If there are not a lot of entries lost here (e.g., because *indiaRDD* is some kind of large dimension table), there is nothing to be gained with this strategy

<h3 id="references"> References </h3>
---	

1. [https://stackoverflow.com/questions/34053302/pyspark-and-broadcast-join-example](https://stackoverflow.com/questions/34053302/pyspark-and-broadcast-join-example)
2. [http://dmtolpeko.com/2015/02/20/map-side-join-in-spark/](http://dmtolpeko.com/2015/02/20/map-side-join-in-spark/)
3. [https://www.facebook.com/notes/facebook-engineering/join-optimization-in-apache-hive/470667928919](https://www.facebook.com/notes/facebook-engineering/join-optimization-in-apache-hive/470667928919)
