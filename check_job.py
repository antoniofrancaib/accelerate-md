#!/usr/bin/env python
"""
Utility script to check the status of GMM experiment jobs.
This is helpful for diagnosing issues with high-dimensional GMM experiments.

Usage:
    python check_job.py --job-id JOB_ID [--tail LINES]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import re
import time

def get_job_status(job_id):
    """Get the status of a job from Slurm."""
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-o", "%T"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            return lines[1]  # First line is header
        return "NOT FOUND"
    except subprocess.CalledProcessError:
        return "ERROR"

def tail_log_file(log_path, lines=50):
    """Show the last N lines of a log file."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    try:
        result = subprocess.run(
            ["tail", "-n", str(lines), log_path],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return "Error reading log file"

def find_gmm_config(job_id):
    """Try to find the GMM config file used for this job."""
    # Look for a slurm script that might reference the config
    slurm_script = Path(f"slurm-{job_id}.out")
    if slurm_script.exists():
        with open(slurm_script, 'r') as f:
            content = f.read()
            # Look for config file reference
            match = re.search(r'--config\s+(\S+)', content)
            if match:
                return match.group(1)
    
    # Look in job output for config references
    stderr_path = Path(f"logs/gmm_experiment_{job_id}.err")
    stdout_path = Path(f"logs/gmm_experiment_{job_id}.out")
    
    for log_path in [stderr_path, stdout_path]:
        if log_path.exists():
            with open(log_path, 'r') as f:
                content = f.read()
                # Look for config file reference
                match = re.search(r'Config:\s+(\S+)', content)
                if match:
                    return match.group(1)
    
    return "Unknown"

def check_for_stuck_job(stdout_path, stderr_path):
    """Check if the job appears to be stuck based on log outputs."""
    if not os.path.exists(stdout_path) and not os.path.exists(stderr_path):
        return "No log files found - job may be queued or not started"
    
    # Check when the logs were last modified
    try:
        stdout_time = os.path.getmtime(stdout_path) if os.path.exists(stdout_path) else 0
        stderr_time = os.path.getmtime(stderr_path) if os.path.exists(stderr_path) else 0
        last_update = max(stdout_time, stderr_time)
        time_since_update = time.time() - last_update
        
        if time_since_update > 3600:  # More than an hour
            return f"Warning: No log updates for {time_since_update/3600:.1f} hours - job may be stuck"
    except Exception as e:
        return f"Error checking file times: {e}"
    
    # Check for common stuck patterns in the logs
    patterns = [
        "Creating dataset",
        "Generating random GMM modes",
        "Starting to sample",
        "Dataset created with",
    ]
    
    last_lines = {}
    for path in [stdout_path, stderr_path]:
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                last_lines[path] = lines[-20:] if len(lines) >= 20 else lines
    
    for pattern in patterns:
        for path, lines in last_lines.items():
            for line in reversed(lines):
                if pattern in line:
                    return f"Job may be stuck at: {line.strip()}"
    
    return "No obvious signs of being stuck"

def main():
    parser = argparse.ArgumentParser(description="Check GMM experiment job status")
    parser.add_argument("--job-id", type=str, required=True, help="Slurm job ID")
    parser.add_argument("--tail", type=int, default=20, help="Number of log lines to show")
    args = parser.parse_args()
    
    job_id = args.job_id
    
    print(f"==== Checking GMM experiment job {job_id} ====")
    
    # Check job status
    status = get_job_status(job_id)
    print(f"Job status: {status}")
    
    # Find config file
    config_file = find_gmm_config(job_id)
    print(f"Config file: {config_file}")
    
    # Check log files
    stdout_path = f"logs/gmm_experiment_{job_id}.out"
    stderr_path = f"logs/gmm_experiment_{job_id}.err"
    
    # Check if job is stuck
    stuck_status = check_for_stuck_job(stdout_path, stderr_path)
    print(f"Status check: {stuck_status}")
    
    # Show tail of logs
    print(f"\n==== Last {args.tail} lines of stdout ====")
    stdout_tail = tail_log_file(stdout_path, args.tail)
    if stdout_tail:
        print(stdout_tail)
    
    print(f"\n==== Last {args.tail} lines of stderr ====")
    stderr_tail = tail_log_file(stderr_path, args.tail)
    if stderr_tail:
        print(stderr_tail)
    
    print("\n==== Recommendations ====")
    if status == "RUNNING" and "stuck" in stuck_status.lower():
        print("- Your job may be stuck. Consider canceling with 'scancel {job_id}' and fixing the issue.")
        print("- For high-dimensional GMMs, try reducing n_mixes or using 'random' mode_arrangement.")
        print("- Check memory usage, as high dimensionality increases memory requirements.")
    elif status == "PENDING":
        print("- Job is still pending. Check resource requirements with 'squeue -j {job_id} -o %all'")
    elif "NOT FOUND" in status:
        print("- Job has completed or was cancelled. Check the log files for any errors.")
    
    print("==== End of report ====")

if __name__ == "__main__":
    main() 