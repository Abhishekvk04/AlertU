# Create comparison data for dlib vs mediapipe
import pandas as pd

# Create comparison data
comparison_data = {
    'Feature': [
        'Installation Ease',
        'Performance (FPS)',
        'Accuracy',
        'Resource Usage',
        'Setup Complexity',
        'Cross-platform',
        'Model Size',
        'Real-time Performance',
        'Robustness',
        'Documentation'
    ],
    'Dlib': [
        'Moderate (requires CMake)',
        '15-20 FPS',
        '95-98%',
        'Medium',
        'Complex (requires .dat file)',
        'Good',
        '99.7 MB',
        'Good',
        'Excellent',
        'Excellent'
    ],
    'MediaPipe': [
        'Easy (pip install)',
        '25-30 FPS',
        '92-96%',
        'Low',
        'Simple',
        'Excellent',
        '2.6 MB',
        'Excellent',
        'Good',
        'Good'
    ],
    'Best Use Case': [
        'MediaPipe',
        'MediaPipe',
        'Dlib',
        'MediaPipe', 
        'MediaPipe',
        'MediaPipe',
        'MediaPipe',
        'MediaPipe',
        'Dlib',
        'Dlib'
    ]
}

df = pd.DataFrame(comparison_data)

# Display the comparison
print("Dlib vs MediaPipe Comparison:")
print("=" * 50)
for index, row in df.iterrows():
    print(f"{row['Feature']:<20} | Dlib: {row['Dlib']:<25} | MediaPipe: {row['MediaPipe']:<25}")

print("\n" + "=" * 50)

# Save to CSV
df.to_csv('drowsiness_detection_comparison.csv', index=False)
print("Comparison saved to drowsiness_detection_comparison.csv")

# Create technical specifications
tech_specs = {
    'Specification': [
        'EAR Threshold',
        'Consecutive Frames',
        'Eye Landmarks (Dlib)',
        'Eye Landmarks (MediaPipe)', 
        'Face Detection Method',
        'Landmark Points',
        'Detection Confidence',
        'Tracking Confidence'
    ],
    'Dlib Implementation': [
        '0.25',
        '20 frames',
        '12 points (6 per eye)',
        'N/A',
        'HOG + Linear SVM',
        '68 facial landmarks',
        'N/A',
        'N/A'
    ],
    'MediaPipe Implementation': [
        '0.25',
        '20 frames', 
        'N/A',
        '12 points (6 per eye)',
        'BlazeFace',
        '468 facial landmarks',
        '0.5',
        '0.5'
    ]
}

tech_df = pd.DataFrame(tech_specs)
print("\nTechnical Specifications:")
print("=" * 50)
for index, row in tech_df.iterrows():
    print(f"{row['Specification']:<25} | Dlib: {row['Dlib Implementation']:<20} | MediaPipe: {row['MediaPipe Implementation']:<20}")

# Save tech specs
tech_df.to_csv('technical_specifications.csv', index=False)
print("\nTechnical specifications saved to technical_specifications.csv")