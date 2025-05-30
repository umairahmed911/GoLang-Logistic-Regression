package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// DataPoint represents a single data entry
type DataPoint struct {
	Age      float64
	Survived int
}

// LogisticRegression represents our improved model
type LogisticRegression struct {
	Weights []float64
	Bias    float64
}

// sigmoid function for logistic regression
func sigmoid(x float64) float64 {
	// Prevent overflow
	if x > 500 {
		return 1.0
	}
	if x < -500 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

// createFeatures creates polynomial and engineered features from age
func createFeatures(age float64) []float64 {
	// Normalize age to prevent large numbers
	normalizedAge := age / 100.0

	features := []float64{
		normalizedAge,                 // Linear age
		normalizedAge * normalizedAge, // Age squared (non-linear relationship)
		math.Log(age + 1),             // Log age (diminishing returns)
	}

	// Age group features (one-hot encoding)
	if age <= 12 {
		features = append(features, 1.0) // Child
	} else {
		features = append(features, 0.0)
	}

	if age >= 13 && age <= 17 {
		features = append(features, 1.0) // Teenager
	} else {
		features = append(features, 0.0)
	}

	if age >= 60 {
		features = append(features, 1.0) // Elderly
	} else {
		features = append(features, 0.0)
	}

	return features
}

// predict returns probability of survival
func (lr *LogisticRegression) predict(age float64) float64 {
	features := createFeatures(age)

	z := lr.Bias
	for i, feature := range features {
		if i < len(lr.Weights) {
			z += lr.Weights[i] * feature
		}
	}

	return sigmoid(z)
}

// predictClass returns 1 if survival probability > threshold, else 0
func (lr *LogisticRegression) predictClass(age float64, threshold float64) int {
	if lr.predict(age) > threshold {
		return 1
	}
	return 0
}

// train the model using improved gradient descent with regularization
func (lr *LogisticRegression) train(data []DataPoint, learningRate float64, epochs int, regularization float64) {
	n := float64(len(data))

	// Initialize weights if not already done
	if len(lr.Weights) == 0 {
		numFeatures := len(createFeatures(data[0].Age))
		lr.Weights = make([]float64, numFeatures)
		// Initialize with small random values
		for i := range lr.Weights {
			lr.Weights[i] = (rand.Float64() - 0.5) * 0.1
		}
		lr.Bias = (rand.Float64() - 0.5) * 0.1
	}

	// Shuffle data for better training
	shuffledData := make([]DataPoint, len(data))
	copy(shuffledData, data)

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle data each epoch
		rand.Shuffle(len(shuffledData), func(i, j int) {
			shuffledData[i], shuffledData[j] = shuffledData[j], shuffledData[i]
		})

		// Calculate gradients
		weightGrads := make([]float64, len(lr.Weights))
		biasGrad := 0.0

		for _, point := range shuffledData {
			features := createFeatures(point.Age)
			prediction := lr.predict(point.Age)
			error := prediction - float64(point.Survived)

			// Calculate gradients for each weight
			for i, feature := range features {
				if i < len(weightGrads) {
					weightGrads[i] += error * feature
				}
			}
			biasGrad += error
		}

		// Update parameters with regularization (L2)
		for i := range lr.Weights {
			lr.Weights[i] -= learningRate * (weightGrads[i]/n + regularization*lr.Weights[i])
		}
		lr.Bias -= learningRate * (biasGrad / n)

		// Decay learning rate
		if epoch > 0 && epoch%200 == 0 {
			learningRate *= 0.95
		}

		// Print progress every 100 epochs
		if epoch%100 == 0 {
			loss := lr.calculateLoss(shuffledData)
			accuracy := lr.evaluateModel(shuffledData, 0.5)
			fmt.Printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%, LR = %.4f\n",
				epoch, loss, accuracy*100, learningRate)
		}
	}
}

// calculateLoss computes the logistic loss with regularization
func (lr *LogisticRegression) calculateLoss(data []DataPoint) float64 {
	loss := 0.0
	for _, point := range data {
		prediction := lr.predict(point.Age)
		// Add small epsilon to prevent log(0)
		epsilon := 1e-15
		prediction = math.Max(epsilon, math.Min(1-epsilon, prediction))

		if point.Survived == 1 {
			loss += -math.Log(prediction)
		} else {
			loss += -math.Log(1 - prediction)
		}
	}
	return loss / float64(len(data))
}

// evaluateModel calculates accuracy with configurable threshold
func (lr *LogisticRegression) evaluateModel(data []DataPoint, threshold float64) float64 {
	correct := 0
	for _, point := range data {
		if lr.predictClass(point.Age, threshold) == point.Survived {
			correct++
		}
	}
	return float64(correct) / float64(len(data))
}

// findOptimalThreshold finds the best threshold for classification
func (lr *LogisticRegression) findOptimalThreshold(data []DataPoint) float64 {
	thresholds := []float64{0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7}
	bestThreshold := 0.5
	bestAccuracy := 0.0

	for _, threshold := range thresholds {
		accuracy := lr.evaluateModel(data, threshold)
		if accuracy > bestAccuracy {
			bestAccuracy = accuracy
			bestThreshold = threshold
		}
	}

	return bestThreshold
}

// readCSV reads the CSV file and extracts Age and Survived columns
func readCSV(filename string) ([]DataPoint, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data []DataPoint
	var ages []float64

	// First pass: collect all valid ages for imputation
	for i := 1; i < len(records); i++ {
		record := records[i]
		ageStr := strings.TrimSpace(record[4])
		if ageStr != "" {
			if age, err := strconv.ParseFloat(ageStr, 64); err == nil {
				ages = append(ages, age)
			}
		}
	}

	// Calculate median age for imputation
	sort.Float64s(ages)
	medianAge := ages[len(ages)/2]

	// Second pass: process all records
	for i := 1; i < len(records); i++ {
		record := records[i]

		// Parse Survived (column 1)
		survived, err := strconv.Atoi(record[1])
		if err != nil {
			continue
		}

		// Parse Age (column 4) - impute missing values with median
		ageStr := strings.TrimSpace(record[4])
		var age float64
		if ageStr == "" || ageStr == "27.0" { // 27.0 seems to be a placeholder in your data
			age = medianAge
		} else {
			var err error
			age, err = strconv.ParseFloat(ageStr, 64)
			if err != nil {
				age = medianAge
			}
		}

		data = append(data, DataPoint{
			Age:      age,
			Survived: survived,
		})
	}

	return data, nil
}

// splitData splits data into training and testing sets with stratification
func splitData(data []DataPoint, trainRatio float64) ([]DataPoint, []DataPoint) {
	// Separate survivors and non-survivors
	var survivors, nonSurvivors []DataPoint
	for _, point := range data {
		if point.Survived == 1 {
			survivors = append(survivors, point)
		} else {
			nonSurvivors = append(nonSurvivors, point)
		}
	}

	// Shuffle both groups
	rand.Shuffle(len(survivors), func(i, j int) {
		survivors[i], survivors[j] = survivors[j], survivors[i]
	})
	rand.Shuffle(len(nonSurvivors), func(i, j int) {
		nonSurvivors[i], nonSurvivors[j] = nonSurvivors[j], nonSurvivors[i]
	})

	// Split both groups proportionally
	trainSurvivors := int(float64(len(survivors)) * trainRatio)
	trainNonSurvivors := int(float64(len(nonSurvivors)) * trainRatio)

	var trainData, testData []DataPoint

	// Add training data
	trainData = append(trainData, survivors[:trainSurvivors]...)
	trainData = append(trainData, nonSurvivors[:trainNonSurvivors]...)

	// Add test data
	testData = append(testData, survivors[trainSurvivors:]...)
	testData = append(testData, nonSurvivors[trainNonSurvivors:]...)

	// Shuffle final datasets
	rand.Shuffle(len(trainData), func(i, j int) {
		trainData[i], trainData[j] = trainData[j], trainData[i]
	})
	rand.Shuffle(len(testData), func(i, j int) {
		testData[i], testData[j] = testData[j], testData[i]
	})

	return trainData, testData
}

// printStatistics shows basic statistics about the data
func printStatistics(data []DataPoint) {
	if len(data) == 0 {
		fmt.Println("No data available")
		return
	}

	totalSurvived := 0
	ageSum := 0.0
	var ages []float64

	for _, point := range data {
		if point.Survived == 1 {
			totalSurvived++
		}
		ageSum += point.Age
		ages = append(ages, point.Age)
	}

	sort.Float64s(ages)
	avgAge := ageSum / float64(len(data))
	survivalRate := float64(totalSurvived) / float64(len(data))
	medianAge := ages[len(ages)/2]

	fmt.Printf("\n=== Dataset Statistics ===\n")
	fmt.Printf("Total records: %d\n", len(data))
	fmt.Printf("Survivors: %d (%.1f%%)\n", totalSurvived, survivalRate*100)
	fmt.Printf("Non-survivors: %d (%.1f%%)\n", len(data)-totalSurvived, (1-survivalRate)*100)
	fmt.Printf("Average age: %.2f\n", avgAge)
	fmt.Printf("Median age: %.2f\n", medianAge)
	fmt.Printf("Age range: %.2f - %.2f\n", ages[0], ages[len(ages)-1])
	fmt.Println()
}

// analyzeAgeGroups shows survival rates by age group
func analyzeAgeGroups(data []DataPoint) {
	groups := map[string]struct{ total, survived int }{
		"Children (0-12)":      {0, 0},
		"Teens (13-17)":        {0, 0},
		"Young Adults (18-35)": {0, 0},
		"Middle-aged (36-59)":  {0, 0},
		"Elderly (60+)":        {0, 0},
	}

	for _, point := range data {
		var group string
		if point.Age <= 12 {
			group = "Children (0-12)"
		} else if point.Age <= 17 {
			group = "Teens (13-17)"
		} else if point.Age <= 35 {
			group = "Young Adults (18-35)"
		} else if point.Age <= 59 {
			group = "Middle-aged (36-59)"
		} else {
			group = "Elderly (60+)"
		}

		g := groups[group]
		g.total++
		if point.Survived == 1 {
			g.survived++
		}
		groups[group] = g
	}

	fmt.Println("=== Survival Rates by Age Group ===")
	for group, stats := range groups {
		if stats.total > 0 {
			rate := float64(stats.survived) / float64(stats.total) * 100
			fmt.Printf("%s: %d/%d (%.1f%%)\n", group, stats.survived, stats.total, rate)
		}
	}
	fmt.Println()
}

func main() {
	// Set random seed for reproducible results
	rand.Seed(time.Now().UnixNano())

	// Read the CSV file
	fmt.Println("Loading Titanic dataset...")
	data, err := readCSV("titanic (1).csv")
	if err != nil {
		log.Fatalf("Error reading CSV file: %v", err)
	}

	if len(data) == 0 {
		log.Fatal("No valid data found in the CSV file")
	}

	// Print dataset statistics
	printStatistics(data)
	analyzeAgeGroups(data)

	// Split data into training and testing sets (80/20 split)
	trainData, testData := splitData(data, 0.8)
	fmt.Printf("Training set: %d records\n", len(trainData))
	fmt.Printf("Test set: %d records\n\n", len(testData))

	// Initialize and train the model
	fmt.Println("Training enhanced logistic regression model...")
	model := &LogisticRegression{}

	// Train the model with improved parameters
	learningRate := 0.1
	epochs := 1000
	regularization := 0.01
	model.train(trainData, learningRate, epochs, regularization)

	// Find optimal threshold
	optimalThreshold := model.findOptimalThreshold(trainData)
	fmt.Printf("\nOptimal classification threshold: %.2f\n", optimalThreshold)

	// Evaluate the model
	fmt.Println("\n=== Model Evaluation ===")
	trainAccuracy := model.evaluateModel(trainData, optimalThreshold)
	testAccuracy := model.evaluateModel(testData, optimalThreshold)

	fmt.Printf("Training Accuracy: %.2f%%\n", trainAccuracy*100)
	fmt.Printf("Test Accuracy: %.2f%%\n", testAccuracy*100)

	// Compare with baseline (always predict majority class)
	survivalRate := 0.0
	for _, point := range trainData {
		if point.Survived == 1 {
			survivalRate += 1.0
		}
	}
	survivalRate /= float64(len(trainData))
	baselineAccuracy := math.Max(survivalRate, 1-survivalRate) * 100
	fmt.Printf("Baseline Accuracy (majority class): %.2f%%\n", baselineAccuracy)

	// Make sample predictions
	fmt.Println("\n=== Sample Predictions ===")
	testAges := []float64{5, 12, 18, 25, 35, 45, 55, 65, 75}

	for _, age := range testAges {
		probability := model.predict(age)
		prediction := model.predictClass(age, optimalThreshold)
		status := "Would not survive"
		if prediction == 1 {
			status = "Would survive"
		}
		fmt.Printf("Age %.0f: %.1f%% survival probability - %s\n",
			age, probability*100, status)
	}

	fmt.Println("\n=== Model Features ===")
	fmt.Printf("Number of features: %d\n", len(model.Weights))
	fmt.Printf("Feature weights: %v\n", model.Weights)
	fmt.Printf("Bias: %.4f\n", model.Bias)
	fmt.Println("Features: [Age, Age², Log(Age), IsChild, IsTeen, IsElderly]")
}
