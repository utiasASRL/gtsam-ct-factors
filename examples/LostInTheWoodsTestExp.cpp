#include "LostInTheWoodsExample.h"

int main(int argc, char* argv[]) {
  int i = 100;
  if (argc > 1) {
    i = atoi(argv[1]);
  }

  // input processing
  string input_file = "LostInTheWoods.txt";

  // load dataset
  DatasetLoader data;
  string filename = findExampleDataFile(input_file);
  data.loadFromFile(filename);
  data.checkSizes();

  Pose2 T_vs(data.d, 0.0, 0.0);
  Key xi(Symbol('x', i));

  // define pose and add to values
  Pose2 pose(data.x_true[i], data.y_true[i], data.th_true[i]);
  cout << "POSE: " << pose << endl;
  Values gt_vals;
  gt_vals.insert(xi, pose);
  for (int j = 0; j < data.n_landmarks; j++) {
    // Check if we have a valid measurement
    if (data.range(i, j) > 0.0) {
      // Get Bearing Range measurement
      BearingRange2 measurement(Rot2(data.bearing(i, j)), data.range(i, j));
      // Get landmark
      Point2 landmark = data.landmarks.row(j);
      // Get prediction expression
      auto predict = BearingRangeLandmarkPrediction(xi, landmark, T_vs);

      // Compute Value and Jacobian
      vector<Matrix> H(1);
      BearingRange2 prediction = predict.value(gt_vals, H);
      // Compute Error
      Vector2 error = prediction.localCoordinates(measurement);

      // Print output
      cout << "LANDMARK " << j << " -------------------- \n Error Value:\n "
           << endl;
      cout << error << endl;
      cout << "Jacobian\n" << H[0] << endl;
    }
  }

  // Jacobian Test
  cout<< "-------- Jacobian Test --------" << endl;
  pose = Pose2(0.0, 0.0, 0.0);
  Point2 landmark(1.0, 0.0);
  auto predict = BearingRangeLandmarkPrediction(xi, landmark, T_vs);
  Values newvals;
  newvals.insert(xi, pose);
  // Compute Value and Jacobian
  vector<Matrix> H(1);
  BearingRange2 prediction = predict.value(newvals, H);
  cout<< "POSE:" << pose << ",  LANDMARK:" << landmark << endl;
  cout << "VALUE:\n";
  prediction.print();
  cout << "JACOBIAN\n" << H[0] << endl;

  pose = Pose2(0.0, 0.0, 3.14159/2);
  landmark = Point2(1.0, data.d);
  predict = BearingRangeLandmarkPrediction(xi, landmark, T_vs);
  Values newvals2;
  newvals2.insert(xi, pose);
  // Compute Value and Jacobian
  prediction = predict.value(newvals2, H);
  cout<< "POSE:" << pose << ",  LANDMARK:" << landmark << endl;
  cout << "VALUE:\n";
  prediction.print();
  cout << "JACOBIAN\n" << H[0] << endl;

  return 0;
}
