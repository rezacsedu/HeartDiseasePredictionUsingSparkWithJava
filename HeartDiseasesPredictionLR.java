package com.example.chapter8.HeartDiseasePrediction;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
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

public class HeartDiseasesPredictionLR {
	public static void main(String[] args) {
		//Create an active Spark session
		SparkSession spark = UtilityForSparkSession.mySession();	
		
		// Taken input and create the RDD from the dataset by specifying the  input source and number of partition. Adjust the number of partition basd on your dataser size
		String input = "heart_diseases/processed_cleveland.data";
		
		long model_building_start = System.currentTimeMillis();
		//String new_data = "heart_diseases/processed_hungarian.data";
		RDD<String> linesRDD = spark.sparkContext().textFile(input, 2);		

		JavaRDD<LabeledPoint> data = linesRDD.toJavaRDD().map(new Function<String, LabeledPoint>() {
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
		
		double[] weights = {0.7, 0.3};
		long split_seed = 12345L;
		JavaRDD<LabeledPoint>[] split = data.randomSplit(weights, split_seed);
		JavaRDD<LabeledPoint> training = split[0];
		JavaRDD<LabeledPoint> test = split[1];	

		//////////////////////////LinearRegressionModel model 0.0% accuracy///////////////////////
		final double stepSize = 0.0000000009;
		final int numberOfIterations = 40; 
		LinearRegressionModel model = LinearRegressionWithSGD.train(JavaRDD.toRDD(training), numberOfIterations, stepSize);
		
		long model_building_end = System.currentTimeMillis();
		System.out.println("Model building time: " + (model_building_end - model_building_start)+" ms");
		
		//Save the model for future use
		
		long model_saving_start = System.currentTimeMillis();
		String model_storage_loc = "models/heartdiseasesLinearRegressionModel";	
		model.save(spark.sparkContext(), model_storage_loc);
		long model_saving_end = System.currentTimeMillis();
		System.out.println("Model saving time: " + (model_saving_end - model_saving_start)+" ms");
		LinearRegressionModel model2 = LinearRegressionModel.load(spark.sparkContext(), model_storage_loc);		
		
		///Evaluate the model
		JavaPairRDD<Double, Double> predictionAndLabel =
				test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
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
			System.out.println("Accuracy of the classification: "+accuracy);			
		spark.stop();
	}
	
}
