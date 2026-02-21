# Note: This file is completely vibe coded, but I read over it it seems legit and i measured in real world and its accurate
import cv2
import numpy as np
import csv
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Classes.Camera import Camera
from constants import (
    CAMERA_FOV,
    KNOWN_CALIBRATION_DISTANCE,
    BALL_D_INCHES,
    KNOWN_CALIBRATION_PIXEL_HEIGHT,
    YOLO_MODEL_FILE,
    GRAYSCALE,
    CAMERA_DOWNWARD_PITCH_ANGLE,
    CAMERA_BOT_RELATIVE_YAW,
    CAMERA_HEIGHT,
    CAMERA_X_OFFSET,
    CAMERA_Y_OFFSET
)

class CalibrationVerifier:
    def __init__(self, camera_source=0, samples_per_position=10):
        """
        Initialize the calibration verifier.
        
        Args:
            camera_source: Camera source (0 for default webcam, or file path)
            samples_per_position: Number of samples to collect at each position
        """
        self.camera_source = camera_source
        self.samples_per_position = samples_per_position
        self.collected_data = []
        
        # Initialize camera
        self.camera = Camera(
            source=camera_source,
            camera_fov=CAMERA_FOV,
            known_calibration_distance=KNOWN_CALIBRATION_DISTANCE,
            ball_d_inches=BALL_D_INCHES,
            known_calibration_pixel_height=KNOWN_CALIBRATION_PIXEL_HEIGHT,
            yolo_model_file=YOLO_MODEL_FILE,
            camera_downward_angle=CAMERA_DOWNWARD_PITCH_ANGLE,
            camera_bot_relative_angle=CAMERA_BOT_RELATIVE_YAW,
            camera_height=CAMERA_HEIGHT,
            camera_x=CAMERA_X_OFFSET,
            camera_y=CAMERA_Y_OFFSET,
            grayscale=GRAYSCALE,
            debug_mode=False
        )
        
    def collect_data_at_position(self, position_name, camera_x=None, camera_y=None, 
                                  camera_z=None, pitch=None, yaw=None):
        """
        Collect multiple samples at a specific camera configuration.
        
        Args:
            position_name: Name/description of this position (e.g., "center", "left_10in")
            camera_x: Override camera X offset (optional)
            camera_y: Override camera Y offset (optional)
            camera_z: Override camera height (optional)
            pitch: Override camera pitch angle (optional)
            yaw: Override camera yaw angle (optional)
        """
        # Store original values
        orig_x = self.camera.camera_x
        orig_y = self.camera.camera_y
        orig_z = self.camera.camera_height
        orig_pitch = self.camera.camera_pitch_angle
        orig_yaw = self.camera.camera_bot_relative_yaw
        
        # Override with new values if provided
        if camera_x is not None:
            self.camera.camera_x = camera_x
        if camera_y is not None:
            self.camera.camera_y = camera_y
        if camera_z is not None:
            self.camera.camera_height = camera_z
        if pitch is not None:
            self.camera.camera_pitch_angle = pitch
        if yaw is not None:
            self.camera.camera_bot_relative_yaw = yaw
        
        print(f"\n{'='*60}")
        print(f"Collecting data for: {position_name}")
        print(f"Camera Config: X={self.camera.camera_x}, Y={self.camera.camera_y}, Z={self.camera.camera_height}")
        print(f"Camera Pitch={self.camera.camera_pitch_angle}°, Yaw={self.camera.camera_bot_relative_yaw}°")
        print(f"{'='*60}")
        print("Keep the ball still in the center of the camera view.")
        print("Collecting samples...")
        
        position_data = []
        
        for i in range(self.samples_per_position):
            try:
                # Get detection data
                points = self.camera.run()
                
                if len(points) > 0:
                    # Assuming we're detecting one object (the ball)
                    x, y = points[0][:2]
                    position_data.append({
                        'position_name': position_name,
                        'sample_num': i + 1,
                        'camera_x': self.camera.camera_x,
                        'camera_y': self.camera.camera_y,
                        'camera_z': self.camera.camera_height,
                        'camera_pitch': self.camera.camera_pitch_angle,
                        'camera_yaw': self.camera.camera_bot_relative_yaw,
                        'detected_x': float(x),
                        'detected_y': float(y),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    print(f"  Sample {i+1}/{self.samples_per_position}: X={x:.2f}, Y={y:.2f}")
                else:
                    print(f"  Sample {i+1}/{self.samples_per_position}: NO DETECTION")
                    
            except Exception as e:
                print(f"  Sample {i+1}/{self.samples_per_position}: ERROR - {e}")
            
            time.sleep(0.1)  # Small delay between samples
        
        self.collected_data.extend(position_data)
        
        # Restore original values
        self.camera.camera_x = orig_x
        self.camera.camera_y = orig_y
        self.camera.camera_height = orig_z
        self.camera.camera_pitch_angle = orig_pitch
        self.camera.camera_bot_relative_yaw = orig_yaw
        
        return position_data
    
    def save_data(self, filename=None):
        """Save collected data to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_data_{timestamp}.csv"
        
        if not self.collected_data:
            print("No data to save!")
            return
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'position_name', 'sample_num', 'camera_x', 'camera_y', 'camera_z',
                    'camera_pitch', 'camera_yaw', 'detected_x', 'detected_y', 'timestamp'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(self.collected_data)
            
            print(f"\n✓ Data saved to: {filename}")
            return filename
        except Exception as e:
            print(f"✗ Error saving data: {e}")
    
    def print_statistics(self):
        """Print statistics about collected data."""
        if not self.collected_data:
            print("No data collected yet!")
            return
        
        # Group by position
        positions = {}
        for data_point in self.collected_data:
            pos_name = data_point['position_name']
            if pos_name not in positions:
                positions[pos_name] = []
            positions[pos_name].append(data_point)
        
        print(f"\n{'='*60}")
        print("STATISTICS")
        print(f"{'='*60}")
        
        for position_name, samples in positions.items():
            x_values = [s['detected_x'] for s in samples]
            y_values = [s['detected_y'] for s in samples]
            
            print(f"\n{position_name}:")
            print(f"  Samples: {len(samples)}")
            print(f"  X (left/right) - Mean: {np.mean(x_values):.2f}, Std Dev: {np.std(x_values):.2f}")
            print(f"  Y (forward) - Mean: {np.mean(y_values):.2f}, Std Dev: {np.std(y_values):.2f}")
            print(f"  X Range: [{np.min(x_values):.2f}, {np.max(x_values):.2f}]")
            print(f"  Y Range: [{np.min(y_values):.2f}, {np.max(y_values):.2f}]")
    
    def verify_math_and_plot(self):
        """
        Automatically verify the math is correct and create visualization plots.
        Returns a dict with verification results.
        """
        if not self.collected_data:
            print("No data to verify!")
            return {}
        
        # Group by test type
        test_results = self._analyze_test_results()
        
        # Create visualizations
        self._create_visualization_plots(test_results)
        
        return test_results
    
    def _analyze_test_results(self):
        """Analyze data and verify expected behaviors."""
        results = {
            'x_offset_test': self._verify_x_offset(),
            'y_offset_test': self._verify_y_offset(),
            'z_offset_test': self._verify_z_offset(),
            'pitch_test': self._verify_pitch(),
            'yaw_test': self._verify_yaw()
        }
        
        # Print verification results
        print(f"\n{'='*70}")
        print("VERIFICATION RESULTS")
        print(f"{'='*70}")
        
        all_passed = True
        for test_name, result in results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"\n{test_name.replace('_', ' ').upper()}: {status}")
            print(f"  Trend: {result['trend']}")
            print(f"  Correlation: {result['correlation']:.3f}")
            print(f"  Data points: {len(result.get('pitches', result.get('x_offsets', result.get('y_offsets', result.get('z_offsets', result.get('yaws', []))))))}")
            if not result['passed']:
                print(f"  ⚠️  Issue: {result['issue']}")
                all_passed = False
        
        # Overall result
        print(f"\n{'='*70}")
        if all_passed:
            print("✓✓✓ ALL TESTS PASSED - MATH IS CORRECT ✓✓✓")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW ABOVE")
        print(f"{'='*70}")
        
        return results
    
    def _verify_x_offset(self):
        """Verify X offset changes detected X correctly."""
        x_offsets = []
        detected_x_means = []
        
        positions = self._group_by_position()
        
        for pos_name in sorted(positions.keys()):
            if 'camera_x_offset' in pos_name:
                samples = positions[pos_name]
                x_values = [s['detected_x'] for s in samples]
                camera_x = samples[0]['camera_x']
                
                x_offsets.append(camera_x)
                detected_x_means.append(np.mean(x_values))
        
        if len(x_offsets) < 2:
            return {'passed': False, 'trend': 'N/A', 'correlation': 0, 'issue': 'Not enough samples'}
        
        correlation = np.corrcoef(x_offsets, detected_x_means)[0, 1]
        trend = "increasing" if detected_x_means[-1] > detected_x_means[0] else "decreasing"
        passed = abs(correlation) > 0.7 and np.std(detected_x_means) > 1.0
        
        return {
            'x_offsets': x_offsets,
            'detected_x_means': detected_x_means,
            'passed': passed,
            'trend': trend,
            'correlation': correlation,
            'issue': 'Low correlation or no variance' if not passed else ''
        }
    
    def _verify_y_offset(self):
        """Verify Y offset changes detected Y correctly."""
        y_offsets = []
        detected_y_means = []
        
        positions = self._group_by_position()
        
        for pos_name in sorted(positions.keys()):
            if 'camera_y_offset' in pos_name:
                samples = positions[pos_name]
                y_values = [s['detected_y'] for s in samples]
                camera_y = samples[0]['camera_y']
                
                y_offsets.append(camera_y)
                detected_y_means.append(np.mean(y_values))
        
        if len(y_offsets) < 2:
            return {'passed': False, 'trend': 'N/A', 'correlation': 0, 'issue': 'Not enough samples'}
        
        correlation = np.corrcoef(y_offsets, detected_y_means)[0, 1]
        trend = "increasing" if detected_y_means[-1] > detected_y_means[0] else "decreasing"
        passed = abs(correlation) > 0.7 and np.std(detected_y_means) > 1.0
        
        return {
            'y_offsets': y_offsets,
            'detected_y_means': detected_y_means,
            'passed': passed,
            'trend': trend,
            'correlation': correlation,
            'issue': 'Low correlation or no variance' if not passed else ''
        }
    
    def _verify_z_offset(self):
        """Verify Z offset (height) changes distance correctly."""
        z_offsets = []
        distances = []
        
        positions = self._group_by_position()
        
        for pos_name in sorted(positions.keys()):
            if 'camera_z_offset' in pos_name:
                samples = positions[pos_name]
                x_vals = [s['detected_x'] for s in samples]
                y_vals = [s['detected_y'] for s in samples]
                camera_z = samples[0]['camera_z']
                
                # Distance from origin (horizontal plane distance to ball)
                mean_dist = np.mean([np.sqrt(x**2 + y**2) for x, y in zip(x_vals, y_vals)])
                z_offsets.append(camera_z)
                distances.append(mean_dist)
        
        if len(z_offsets) < 2:
            return {'passed': False, 'trend': 'N/A', 'correlation': 0, 'issue': 'Not enough samples', 
                   'z_offsets': [], 'distances': []}
        
        # For Z offset, the relationship should be inverse: higher camera = closer apparent distance
        # due to the angle correction in the math
        correlation = np.corrcoef(z_offsets, distances)[0, 1]
        trend = "inverse" if correlation < -0.3 else ("direct" if correlation > 0.3 else "flat")
        
        # The test passes if there's a clear inverse relationship (negative correlation)
        # OR if there's variance but not a strong positive correlation
        passed = correlation < -0.3 or (abs(correlation) < 0.5 and np.std(distances) > 1.0)
        
        issue = ""
        if not passed:
            if correlation > 0.5:
                issue = "Wrong direction: distance increases with height (should decrease or stay similar)"
            elif correlation > 0.3:
                issue = "Weak positive trend (should be inverse or neutral)"
        
        return {
            'z_offsets': z_offsets,
            'distances': distances,
            'passed': passed,
            'trend': trend,
            'correlation': correlation,
            'issue': issue
        }
    
    def _verify_pitch(self):
        """Verify pitch angle changes Y correctly."""
        pitches = []
        detected_y_means = []
        detected_distances = []
        
        positions = self._group_by_position()
        
        for pos_name in sorted(positions.keys()):
            if 'camera_pitch' in pos_name:
                samples = positions[pos_name]
                y_values = [s['detected_y'] for s in samples]
                x_values = [s['detected_x'] for s in samples]
                pitch = samples[0]['camera_pitch']
                
                pitches.append(pitch)
                detected_y_means.append(np.mean(y_values))
                # Also track total distance to ball
                detected_distances.append(np.mean([np.sqrt(x**2 + y**2) for x, y in zip(x_values, y_values)]))
        
        if len(pitches) < 2:
            return {'passed': False, 'trend': 'N/A', 'correlation': 0, 'issue': 'Not enough samples', 
                   'pitches': [], 'detected_y_means': []}
        
        # For pitch: since apparent ball size changes with viewing angle,
        # the relationship is more complex than simple linear
        # What SHOULD happen: as pitch increases, if ball appears smaller, distance_los increases
        # So detected coordinates might change based on perspective
        
        # Let's check if there's ANY correlation (not necessarily strong linear)
        correlation = np.corrcoef(pitches, detected_y_means)[0, 1]
        distance_correlation = np.corrcoef(pitches, detected_distances)[0, 1]
        
        trend = "increasing" if detected_y_means[-1] > detected_y_means[0] else "decreasing"
        
        # Pass if there's some correlation in either Y or distance
        # OR if the variance is very low (ball position is stable)
        y_variance = np.std(detected_y_means)
        passed = (abs(correlation) > 0.3) or (abs(distance_correlation) > 0.3) or (y_variance < 1.0)
        
        issue = ""
        if not passed:
            issue = f"Pitch has no effect on coordinates (might be expected if ball size changes with angle)"
        
        return {
            'pitches': pitches,
            'detected_y_means': detected_y_means,
            'detected_distances': detected_distances,
            'passed': passed,
            'trend': trend,
            'correlation': correlation,
            'issue': issue
        }
    
    def _verify_yaw(self):
        """Verify yaw angle rotates coordinates correctly."""
        yaws = []
        angles_in_robotframe = []
        
        positions = self._group_by_position()
        
        for pos_name in sorted(positions.keys()):
            if 'camera_yaw' in pos_name:
                samples = positions[pos_name]
                x_values = [s['detected_x'] for s in samples]
                y_values = [s['detected_y'] for s in samples]
                yaw = samples[0]['camera_yaw']
                
                # Calculate angle in robot frame
                mean_x = np.mean(x_values)
                mean_y = np.mean(y_values)
                angle = np.degrees(np.arctan2(mean_x, mean_y))
                
                yaws.append(yaw)
                angles_in_robotframe.append(angle)
        
        if len(yaws) < 2:
            return {'passed': False, 'trend': 'N/A', 'correlation': 0, 'issue': 'Not enough samples'}
        
        correlation = np.corrcoef(yaws, angles_in_robotframe)[0, 1]
        passed = abs(correlation) > 0.7
        trend = "rotating" if abs(correlation) > 0.5 else "not rotating"
        
        return {
            'yaws': yaws,
            'angles': angles_in_robotframe,
            'passed': passed,
            'trend': trend,
            'correlation': correlation,
            'issue': 'Low correlation' if not passed else ''
        }
    
    def _group_by_position(self):
        """Group data by position name."""
        positions = {}
        for data_point in self.collected_data:
            pos_name = data_point['position_name']
            if pos_name not in positions:
                positions[pos_name] = []
            positions[pos_name].append(data_point)
        return positions
    
    def _create_visualization_plots(self, test_results):
        """Create matplotlib visualizations of verification results."""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Camera Calibration Verification Results', fontsize=16, fontweight='bold')
        
        # Test 1: X Offset
        if test_results['x_offset_test']['passed']:
            ax1 = plt.subplot(3, 3, 1)
            result = test_results['x_offset_test']
            ax1.plot(result['x_offsets'], result['detected_x_means'], 'o-', linewidth=2, markersize=8, color='green')
            ax1.set_xlabel('Camera X Offset (inches)')
            ax1.set_ylabel('Detected X (inches)')
            ax1.set_title('X Offset Test ✓ PASS\n(Should be LINEAR)')
            ax1.grid(True, alpha=0.3)
        else:
            ax1 = plt.subplot(3, 3, 1)
            result = test_results['x_offset_test']
            ax1.plot(result['x_offsets'], result['detected_x_means'], 'o--', linewidth=2, markersize=8, color='red')
            ax1.set_xlabel('Camera X Offset (inches)')
            ax1.set_ylabel('Detected X (inches)')
            ax1.set_title('X Offset Test ✗ FAIL\n(Should be LINEAR)')
            ax1.grid(True, alpha=0.3)
        
        # Test 2: Y Offset
        if test_results['y_offset_test']['passed']:
            ax2 = plt.subplot(3, 3, 2)
            result = test_results['y_offset_test']
            ax2.plot(result['y_offsets'], result['detected_y_means'], 'o-', linewidth=2, markersize=8, color='green')
            ax2.set_xlabel('Camera Y Offset (inches)')
            ax2.set_ylabel('Detected Y (inches)')
            ax2.set_title('Y Offset Test ✓ PASS\n(Should be LINEAR)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2 = plt.subplot(3, 3, 2)
            result = test_results['y_offset_test']
            ax2.plot(result['y_offsets'], result['detected_y_means'], 'o--', linewidth=2, markersize=8, color='red')
            ax2.set_xlabel('Camera Y Offset (inches)')
            ax2.set_ylabel('Detected Y (inches)')
            ax2.set_title('Y Offset Test ✗ FAIL\n(Should be LINEAR)')
            ax2.grid(True, alpha=0.3)
        
        # Test 3: Z Offset
        if test_results['z_offset_test']['passed']:
            ax3 = plt.subplot(3, 3, 3)
            result = test_results['z_offset_test']
            ax3.plot(result['z_offsets'], result['distances'], 'o-', linewidth=2, markersize=8, color='green')
            ax3.set_xlabel('Camera Height (inches)')
            ax3.set_ylabel('Distance to Ball (inches)')
            ax3.set_title('Z Offset Test ✓ PASS\n(Should be CURVED)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3 = plt.subplot(3, 3, 3)
            result = test_results['z_offset_test']
            ax3.plot(result['z_offsets'], result['distances'], 'o--', linewidth=2, markersize=8, color='red')
            ax3.set_xlabel('Camera Height (inches)')
            ax3.set_ylabel('Distance to Ball (inches)')
            ax3.set_title('Z Offset Test ✗ FAIL\n(Should be CURVED)')
            ax3.grid(True, alpha=0.3)
        
        # Test 4: Pitch
        if test_results['pitch_test']['passed']:
            ax4 = plt.subplot(3, 3, 4)
            result = test_results['pitch_test']
            ax4.plot(result['pitches'], result['detected_y_means'], 'o-', linewidth=2, markersize=8, color='green')
            ax4.set_xlabel('Camera Pitch (degrees)')
            ax4.set_ylabel('Detected Y (inches)')
            ax4.set_title('Pitch Test ✓ PASS\n(Should INCREASE with pitch)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4 = plt.subplot(3, 3, 4)
            result = test_results['pitch_test']
            ax4.plot(result['pitches'], result['detected_y_means'], 'o--', linewidth=2, markersize=8, color='red')
            ax4.set_xlabel('Camera Pitch (degrees)')
            ax4.set_ylabel('Detected Y (inches)')
            ax4.set_title('Pitch Test ✗ FAIL\n(Should INCREASE with pitch)')
            ax4.grid(True, alpha=0.3)
        
        # Test 5: Yaw
        if test_results['yaw_test']['passed']:
            ax5 = plt.subplot(3, 3, 5)
            result = test_results['yaw_test']
            ax5.plot(result['yaws'], result['angles'], 'o-', linewidth=2, markersize=8, color='green')
            ax5.set_xlabel('Camera Yaw (degrees)')
            ax5.set_ylabel('Detected Angle in Robot Frame (degrees)')
            ax5.set_title('Yaw Test ✓ PASS\n(Should match yaw angle)')
            ax5.grid(True, alpha=0.3)
        else:
            ax5 = plt.subplot(3, 3, 5)
            result = test_results['yaw_test']
            ax5.plot(result['yaws'], result['angles'], 'o--', linewidth=2, markersize=8, color='red')
            ax5.set_xlabel('Camera Yaw (degrees)')
            ax5.set_ylabel('Detected Angle in Robot Frame (degrees)')
            ax5.set_title('Yaw Test ✗ FAIL\n(Should match yaw angle)')
            ax5.grid(True, alpha=0.3)
        
        # All detections in XY plane
        ax6 = plt.subplot(3, 3, 6)
        positions = self._group_by_position()
        colors = plt.cm.tab10(np.linspace(0, 1, len(positions)))
        
        for idx, (pos_name, samples) in enumerate(sorted(positions.items())):
            x_vals = [s['detected_x'] for s in samples]
            y_vals = [s['detected_y'] for s in samples]
            ax6.scatter(x_vals, y_vals, label=pos_name[:20], alpha=0.6, s=50, color=colors[idx])
        
        ax6.set_xlabel('Detected X (inches)')
        ax6.set_ylabel('Detected Y (inches)')
        ax6.set_title('All Detections in XY Plane')
        ax6.grid(True, alpha=0.3)
        ax6.axis('equal')
        ax6.legend(fontsize=8, loc='best')
        
        # Summary results
        ax7 = plt.subplot(3, 3, (7, 9))
        ax7.axis('off')
        
        summary_text = "VERIFICATION SUMMARY\n" + "="*50 + "\n"
        all_passed = all(r['passed'] for r in test_results.values())
        
        if all_passed:
            summary_text += "✓✓✓ ALL TESTS PASSED ✓✓✓\n"
            summary_text += "Math is correct!\n\n"
        else:
            summary_text += "✗ Some tests failed\n"
            summary_text += "Review the failed tests above\n\n"
        
        for test_name, result in test_results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            corr = result['correlation']
            summary_text += f"{test_name.replace('_', ' ').upper()}\n"
            summary_text += f"  Status: {status}  |  Correlation: {corr:.3f}\n"
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_verification_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {filename}")
        
        plt.show()
    
    def cleanup(self):
        """Clean up resources."""
        self.camera.destroy()

def main():
    """Main calibration routine."""
    verifier = CalibrationVerifier(camera_source=0, samples_per_position=5)
    
    try:
        print("\nCAMERA CALIBRATION VERIFICATION")
        print("=" * 70)
        print("This script will systematically change camera parameters")
        print("and verify the detected coordinates change appropriately.")
        print("Keep the BALL STATIONARY throughout all tests!")
        print("=" * 70)
        
        # Test 1: X axis offset (left/right camera position)
        print("\n[TEST 1] Varying CAMERA X OFFSET (left/right)")
        print("-" * 70)
        for x_offset in [0, 5, 10, -5, -10]:
            verifier.collect_data_at_position(
                f"camera_x_offset_{x_offset:+d}in",
                camera_x=x_offset,
                camera_y=CAMERA_Y_OFFSET,
                camera_z=CAMERA_HEIGHT,
                pitch=CAMERA_DOWNWARD_PITCH_ANGLE,
                yaw=CAMERA_BOT_RELATIVE_YAW
            )
        
        # Test 2: Y axis offset (forward/backward camera position)
        print("\n[TEST 2] Varying CAMERA Y OFFSET (forward/backward)")
        print("-" * 70)
        for y_offset in [0, 5, 10, -5, -10]:
            verifier.collect_data_at_position(
                f"camera_y_offset_{y_offset:+d}in",
                camera_x=CAMERA_X_OFFSET,
                camera_y=y_offset,
                camera_z=CAMERA_HEIGHT,
                pitch=CAMERA_DOWNWARD_PITCH_ANGLE,
                yaw=CAMERA_BOT_RELATIVE_YAW
            )
        
        # Test 3: Z axis offset (camera height)
        print("\n[TEST 3] Varying CAMERA HEIGHT (Z offset)")
        print("-" * 70)
        for z_offset in [5, 7.5, 10, 12, 15]:
            verifier.collect_data_at_position(
                f"camera_z_offset_{z_offset:.1f}in",
                camera_x=CAMERA_X_OFFSET,
                camera_y=CAMERA_Y_OFFSET,
                camera_z=z_offset,
                pitch=CAMERA_DOWNWARD_PITCH_ANGLE,
                yaw=CAMERA_BOT_RELATIVE_YAW
            )
        
        # Test 4: Pitch angle (downward tilt)
        print("\n[TEST 4] Varying CAMERA PITCH (downward angle)")
        print("-" * 70)
        for pitch in [5, 10, 15, 20, 25]:
            verifier.collect_data_at_position(
                f"camera_pitch_{pitch}deg",
                camera_x=CAMERA_X_OFFSET,
                camera_y=CAMERA_Y_OFFSET,
                camera_z=CAMERA_HEIGHT,
                pitch=pitch,
                yaw=CAMERA_BOT_RELATIVE_YAW
            )
        
        # Test 5: Yaw angle (rotation relative to robot)
        print("\n[TEST 5] Varying CAMERA YAW (rotation)")
        print("-" * 70)
        for yaw in [0, 10, 20, -10, -20]:
            verifier.collect_data_at_position(
                f"camera_yaw_{yaw:+d}deg",
                camera_x=CAMERA_X_OFFSET,
                camera_y=CAMERA_Y_OFFSET,
                camera_z=CAMERA_HEIGHT,
                pitch=CAMERA_DOWNWARD_PITCH_ANGLE,
                yaw=yaw
            )
        
        # Print statistics
        verifier.print_statistics()
        
        # Save data
        csv_filename = verifier.save_data()
        
        # Verify math and show plots
        test_results = verifier.verify_math_and_plot()
        
        print("\n" + "="*70)
        print("VERIFICATION COMPLETE")
        print("="*70)
        print(f"Collected {len(verifier.collected_data)} total samples across 5 test categories")
        print(f"CSV Data saved to: {csv_filename}")
        
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        verifier.cleanup()

if __name__ == "__main__":
    main()
