package com.example.chapter8.HeartDiseasePrediction;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

/*
 * 
 * 
 * @Author: Md. Rezaul Karim
 * @Date: 16/08/2016
 * 
 * 
 */

public class HeartDiseasesPredictionNB {
	public static void main(String[] args) {
		// Create an active Spark session
		SparkSession spark = UtilityForSparkSession.mySession();

		// Taken input and create the RDD from the dataset by specifying the
		// input source and number of partition. Adjust the number of partition
		// basd on your dataser size
		
		long model_building_start = System.currentTimeMillis();
		String input = "heart_diseases/processed_cleveland.data";
		//String new_data = "heart_diseases/processed_hungarian.data";
		RDD<String> linesRDD = spark.sparkContext().textFile(input, 2);

		// For the new data
		/*
		 * RDD<String> linesRDD = spark.sparkContext().textFile(new_data, 2);
		 */

		JavaRDD<LabeledPoint> data = linesRDD.toJavaRDD().map(new Function<String, LabeledPoint>() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public LabeledPoint call(String row) throws Exception {
				String line = row.replaceAll("\\?", "999999.0");
				String[] tokens = line.split(",");
				Integer last = Integer.parseInt(tokens[13]);
				double[] features = new double[13];
				for (int i = 0; i < 13; i++) {
					features[i] = Double.parseDouble(tokens[i]);
				}
				Vector v = new DenseVector(features);
				Double value = 0.0;
				if (last.intValue() > 0)
					value = 1.0;
				LabeledPoint lp = new LabeledPoint(value, v);
				return lp;
			}
		});

		double[] weights = { 0.7, 0.3 };
		long split_seed = 12345L;
		JavaRDD<LabeledPoint>[] split = data.randomSplit(weights, split_seed);
		JavaRDD<LabeledPoint> training = split[0];
		JavaRDD<LabeledPoint> test = split[1];

		/////////////////////// Naive Bayes 45% ///////////////////////
		final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0, "multinomial");
		long model_building_end = System.currentTimeMillis();
		System.out.println("Model building time: " + (model_building_end - model_building_start)+" ms");
		
		long model_saving_start = System.currentTimeMillis();
		String modelStorageLoc = "models/heartdiseaseNaiveBayesModel";
		model.save(spark.sparkContext(), modelStorageLoc );
		long model_saving_end = System.currentTimeMillis();
		System.out.println("Model saving time: " + (model_saving_end - model_saving_start)+" ms");
		
		NaiveBayesModel model2 = NaiveBayesModel.load(spark.sparkContext(), modelStorageLoc);

		/// Evaluate the model
		JavaPairRDD<Double, Double> predictionAndLabel = test
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<>(model2.predict(p.features()), p.label());
					}
				});

		double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
			@Override
			public Boolean call(Tuple2<Double, Double> pl) {
				return pl._1().equals(pl._2());
			}
		}).count() / (double) test.count();
		System.out.println("Accuracy of the classification: " + accuracy);
		spark.stop();
	}

}
