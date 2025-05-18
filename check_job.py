#!/usr/bin/env python
"""
Utility script to check the status of a running job.
This helps diagnose jobs that appear to be hanging.

Usage:
    python check_job.py JOB_ID

Example:
    python check_job.py 9517188
"""

import sys
import subprocess
import os
from pathlib import Path
import re

def run_cmd(cmd):
    """Run a shell command and return output as string."""
    return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()

def check_gpu_usage(job_id):
    """Check if the job is using GPU."""
    try:
        # Get the node where the job is running
        node_info = run_cmd(f"squeue -j {job_id} -o %N")
        
        # Check if we have an actual node or just the header
        if "NODELIST" in node_info:
            lines = node_info.split('\n')
            if len(lines) <= 1:
                print("Job is pending - no node assigned yet")
                return
            node = lines[1]
        else:
            # Sometimes squeue doesn't return the header
            if node_info == "(Priority)" or node_info == "(Resources)" or node_info == "":
                print("Job is pending - no node assigned yet")
                return
            node = node_info
        
        print(f"Job is running on node: {node}")
        
        # Check if we can ssh to the node
        ssh_test = run_cmd(f"ssh {node} echo 'OK' 2>/dev/null || echo 'Failed'")
        if ssh_test == "Failed":
            print("Cannot connect to node directly, skipping GPU check")
            return
            
        # Check GPU usage on that node for your user
        gpu_usage = run_cmd(f"ssh {node} 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv'")
        print("GPU Usage:")
        print(gpu_usage)
        
        # Get PIDs for the job
        job_pids = run_cmd(f"ssh {node} 'ps -u $USER -o pid,ppid,pcpu,pmem,cmd | grep {job_id}'")
        print("\nProcesses for this job:")
        print(job_pids)
        
    except subprocess.CalledProcessError:
        print("Could not check GPU usage")
    except Exception as e:
        print(f"Error checking GPU usage: {str(e)}")

def check_log_progress(job_id):
    """Check the progress by analyzing job output files."""
    log_files = list(Path("logs").glob(f"*{job_id}*"))
    if not log_files:
        print("No log files found for this job ID")
        return
        
    print(f"\nFound {len(log_files)} log files: {[f.name for f in log_files]}")
    
    # Check each log file
    for log_file in log_files:
        if log_file.stat().st_size == 0:
            print(f"{log_file} is empty")
            continue
            
        print(f"\nAnalyzing {log_file}:")
        try:
            # Read last 20 lines for a quick overview
            last_lines = run_cmd(f"tail -n 20 {log_file}")
            print("\nLast few lines:")
            print(last_lines)
            
            # Check for debug messages that indicate progress
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for debug messages
            debug_msgs = re.findall(r'\[DEBUG.*?\].*', content)
            if debug_msgs:
                print(f"\nFound {len(debug_msgs)} debug messages")
                print("Last 5 debug messages:")
                for msg in debug_msgs[-5:]:
                    print(msg)
                
                # If job is in training phase, check epoch progress
                epoch_msgs = re.findall(r'\[DEBUG.*?\] Epoch (\d+)', content)
                if epoch_msgs:
                    last_epoch = epoch_msgs[-1]
                    print(f"\nTraining progress: Last epoch is {last_epoch}")
                
                # Check if job is in evaluation phase
                eval_msgs = [msg for msg in debug_msgs if "evaluation" in msg.lower()]
                if eval_msgs:
                    print("\nJob reached evaluation phase:")
                    for msg in eval_msgs:
                        print(msg)
        except Exception as e:
            print(f"Error analyzing log file: {e}")

def check_output_files(job_id):
    """Check if the job has created any output files."""
    try:
        # Look for output directories that might have been created
        output_dirs = list(Path("outputs").glob("*"))
        if not output_dirs:
            print("\nNo output directories found")
            return
            
        print("\nOutput directories:")
        for dir_path in output_dirs:
            if dir_path.is_dir():
                files = list(dir_path.glob("**/*"))
                file_count = len(files)
                dir_size = sum(f.stat().st_size for f in files if f.is_file())
                last_modified = max([f.stat().st_mtime for f in files], default=0) if files else 0
                
                print(f"- {dir_path}: {file_count} files, {dir_size/1024/1024:.2f} MB, " + 
                      (f"last modified: {run_cmd(f'date -d @{int(last_modified)}')})" if last_modified else "empty"))
                
                # List a few sample files
                if files:
                    print("  Sample files:")
                    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                        print(f"  - {f.relative_to(dir_path)}: {f.stat().st_size/1024:.1f} KB")
    except Exception as e:
        print(f"Error checking output files: {e}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} JOB_ID")
        sys.exit(1)
        
    job_id = sys.argv[1]
    
    print(f"Checking status of job {job_id}")
    
    # Check if job is running
    try:
        job_status = run_cmd(f"squeue -j {job_id} -o %T")
        if "STAT" in job_status:  # Header row
            lines = job_status.split('\n')
            if len(lines) <= 1:
                print("Job not found - it may have completed or been cancelled")
                status = "NOT_FOUND"
            else:
                status = lines[1]
        else:
            status = job_status
            
        print(f"Job status: {status}")
        
        if status not in ["RUNNING", "PENDING", "COMPLETING"]:
            print(f"Job is not currently running (status: {status})")
            
            # For completed jobs, check the output
            if status == "NOT_FOUND":
                print("Checking for job output files...")
    except Exception as e:
        print(f"Error checking job status: {str(e)}")
        status = "ERROR"
        
    # Check GPU usage if job is running
    if status in ["RUNNING", "COMPLETING"]:
        check_gpu_usage(job_id)
    
    # Check logs (useful even if job is not running anymore)
    check_log_progress(job_id)
    
    # Check output files
    check_output_files(job_id)

if __name__ == "__main__":
    main() 