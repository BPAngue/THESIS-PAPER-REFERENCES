"""
Comparison test script for old vs new optimization approaches
"""
import subprocess
import time
import os
from datetime import datetime

def run_optimization(script_name, num_runs=5):
    """Run an optimization script multiple times and collect statistics"""
    results = []
    
    print(f"\n{'='*70}")
    print(f"Testing: {script_name}")
    print(f"{'='*70}\n")
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                ['python', script_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per run
            )
            
            elapsed = time.time() - start_time
            
            # Parse output for correctness
            output = result.stdout
            correctness = None
            gates = None
            
            # Look for "Correct outputs: X/Y" pattern
            for line in output.split('\n'):
                if 'Correctness:' in line or 'Correct outputs:' in line:
                    try:
                        parts = line.split('/')
                        if len(parts) == 2:
                            correct = int(parts[0].split()[-1])
                            total = int(parts[1].split()[0])
                            correctness = (correct, total)
                    except:
                        pass
                
                if 'Active gates' in line or 'Active Gates Used:' in line:
                    try:
                        gates = int(line.split(':')[-1].strip().split()[0])
                    except:
                        pass
            
            results.append({
                'run': run + 1,
                'correctness': correctness,
                'gates': gates,
                'time': elapsed,
                'success': correctness[0] == correctness[1] if correctness else False
            })
            
            print(f"  Correctness: {correctness[0]}/{correctness[1] if correctness else 'N/A'}")
            print(f"  Time: {elapsed:.1f}s")
            
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>300s)")
            results.append({
                'run': run + 1,
                'correctness': None,
                'gates': None,
                'time': 300,
                'success': False
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'run': run + 1,
                'correctness': None,
                'gates': None,
                'time': 0,
                'success': False
            })
    
    return results

def analyze_results(results, script_name):
    """Analyze and print statistics"""
    print(f"\n{'='*70}")
    print(f"Results for: {script_name}")
    print(f"{'='*70}\n")
    
    # Success rate
    successes = sum(1 for r in results if r['success'])
    total = len(results)
    success_rate = 100 * successes / total if total > 0 else 0
    
    print(f"Success Rate: {successes}/{total} ({success_rate:.1f}%)")
    
    # Correctness statistics
    correctness_scores = []
    for r in results:
        if r['correctness']:
            score = 100 * r['correctness'][0] / r['correctness'][1]
            correctness_scores.append(score)
    
    if correctness_scores:
        avg_correctness = sum(correctness_scores) / len(correctness_scores)
        min_correctness = min(correctness_scores)
        max_correctness = max(correctness_scores)
        print(f"Correctness: {avg_correctness:.1f}% (min: {min_correctness:.1f}%, max: {max_correctness:.1f}%)")
    
    # Gates statistics (for successful runs)
    gate_counts = [r['gates'] for r in results if r['success'] and r['gates']]
    if gate_counts:
        avg_gates = sum(gate_counts) / len(gate_counts)
        min_gates = min(gate_counts)
        max_gates = max(gate_counts)
        print(f"Gates (successful runs): {avg_gates:.1f} (min: {min_gates}, max: {max_gates})")
    
    # Time statistics
    times = [r['time'] for r in results if r['time'] > 0]
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"Time: {avg_time:.1f}s (min: {min_time:.1f}s, max: {max_time:.1f}s)")
    
    return {
        'success_rate': success_rate,
        'avg_correctness': avg_correctness if correctness_scores else 0,
        'avg_gates': avg_gates if gate_counts else None,
        'avg_time': avg_time if times else 0
    }

def compare_approaches(old_stats, new_stats):
    """Compare old vs new approach"""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<30} {'Old Approach':<20} {'New Approach':<20} {'Improvement'}")
    print("-" * 80)
    
    # Success rate
    old_sr = old_stats['success_rate']
    new_sr = new_stats['success_rate']
    sr_diff = new_sr - old_sr
    sr_symbol = "↑" if sr_diff > 0 else "↓" if sr_diff < 0 else "="
    print(f"{'Success Rate':<30} {old_sr:>6.1f}% {new_sr:>19.1f}% {sr_symbol:>7} {abs(sr_diff):>6.1f}%")
    
    # Correctness
    old_corr = old_stats['avg_correctness']
    new_corr = new_stats['avg_correctness']
    corr_diff = new_corr - old_corr
    corr_symbol = "↑" if corr_diff > 0 else "↓" if corr_diff < 0 else "="
    print(f"{'Avg Correctness':<30} {old_corr:>6.1f}% {new_corr:>19.1f}% {corr_symbol:>7} {abs(corr_diff):>6.1f}%")
    
    # Gates (if available)
    if old_stats['avg_gates'] and new_stats['avg_gates']:
        old_gates = old_stats['avg_gates']
        new_gates = new_stats['avg_gates']
        gates_diff = new_gates - old_gates
        gates_symbol = "↓" if gates_diff < 0 else "↑" if gates_diff > 0 else "="
        print(f"{'Avg Gates (when successful)':<30} {old_gates:>6.1f} {new_gates:>19.1f} {gates_symbol:>7} {abs(gates_diff):>6.1f}")
    
    # Time
    old_time = old_stats['avg_time']
    new_time = new_stats['avg_time']
    time_diff = new_time - old_time
    time_symbol = "↓" if time_diff < 0 else "↑" if time_diff > 0 else "="
    speedup = old_time / new_time if new_time > 0 else 0
    print(f"{'Avg Time':<30} {old_time:>6.1f}s {new_time:>19.1f}s {time_symbol:>7} {abs(time_diff):>6.1f}s")
    if speedup > 0:
        print(f"{'Speedup':<30} {'':<20} {speedup:>6.2f}x")
    
    print("\n" + "="*80)
    
    # Recommendation
    if new_sr > old_sr and new_corr >= old_corr:
        print("\n✓ RECOMMENDATION: New approach shows clear improvement!")
    elif new_sr == old_sr and new_time < old_time:
        print("\n→ RECOMMENDATION: Similar success but faster - consider new approach")
    else:
        print("\n→ RECOMMENDATION: Results mixed - may need parameter tuning")

if __name__ == "__main__":
    print("="*70)
    print("OPTIMIZATION COMPARISON TEST")
    print("="*70)
    print("\nThis script will run both optimization approaches multiple times")
    print("and compare their performance.")
    print("\nNOTE: Each run may take several minutes. Total time: ~30-60 minutes")
    print("\nPress Ctrl+C to cancel\n")
    
    num_runs = int(input("Number of runs per approach (recommended: 5-10): ") or "5")
    
    # Check if files exist
    old_script = "hybrid_multithreaded.py"
    new_script = "improved_pipeline.py"
    
    if not os.path.exists(old_script):
        print(f"\nError: {old_script} not found!")
        exit(1)
    
    if not os.path.exists(new_script):
        print(f"\nError: {new_script} not found!")
        exit(1)
    
    # Run tests
    print("\n" + "="*70)
    print("PHASE 1: Testing OLD approach")
    print("="*70)
    old_results = run_optimization(old_script, num_runs)
    old_stats = analyze_results(old_results, old_script)
    
    print("\n" + "="*70)
    print("PHASE 2: Testing NEW approach")
    print("="*70)
    new_results = run_optimization(new_script, num_runs)
    new_stats = analyze_results(new_results, new_script)
    
    # Compare
    compare_approaches(old_stats, new_stats)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comparison_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("DETAILED COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Runs per approach: {num_runs}\n\n")
        
        f.write("OLD APPROACH RESULTS:\n")
        for r in old_results:
            f.write(f"  Run {r['run']}: Correctness={r['correctness']}, "
                   f"Gates={r['gates']}, Time={r['time']:.1f}s, "
                   f"Success={r['success']}\n")
        
        f.write("\nNEW APPROACH RESULTS:\n")
        for r in new_results:
            f.write(f"  Run {r['run']}: Correctness={r['correctness']}, "
                   f"Gates={r['gates']}, Time={r['time']:.1f}s, "
                   f"Success={r['success']}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
