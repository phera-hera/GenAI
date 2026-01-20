Option 1: Feature Branch (BEST PRACTICE) ✅                                                      
                                                                                                   
  # 1. Make sure dev is clean and pushed                                                           
  git status                                                                                       
  git push origin dev                                                                              
                                                                                                   
  # 2. Merge dev into main                                                                         
  git checkout main                                                                                
  git pull origin main                                                                             
  git merge dev                                                                                    
  git push origin main                                                                             
                                                                                                   
  # 3. Create new feature branch from dev for next work                                            
  git checkout dev                                                                                 
  git pull origin dev                                                                              
  git checkout -b feature/chunking-optimization                                                    
  # Now develop here                                                                               
                                                                                                   
  # 4. When done, create PR to merge back to dev                                                   
  git push origin feature/chunking-optimization                                                    
  # Then create PR in GitHub: feature/chunking-optimization → dev                                  
                                                                                                   
  Option 2: Continue on dev directly (SIMPLER but less organized)                                  
                                                                                                   
  git checkout dev                                                                                 
  git pull origin dev                                                                              
  # Continue developing on dev                                                                     
  # When ready, merge dev→main                                                                     
                                                                                                   
  ---                                                                                              
  Do You Need to Restart the App?                                                                  
                                                                                                   
  Short answer: NO ❌                                                                              
                                                                                                   
  When you merge branches on GitHub, your running app doesn't care:                                
  - If API is running from code in /dev folder → keeps running                                     
  - If Streamlit is running from code in /dev folder → keeps running                               
  - Git merges don't affect already-running processes                                              
                                                                                                   
  You only restart if:                                                                             
  - You change .env file                                                                           
  - You update dependencies (requirements/pyproject.toml)                                          
  - You want to reload code changes (but hot-reload usually handles this)                          
                                                                                                   
  ---                                                                                              
  My Recommendation for You                                                                        
                                                                                                   
  Use Option 1 (Feature Branches):                                                                 
                                                                                                   
  1. Merge dev → main (save current work):                                                         
  git checkout main                                                                                
  git merge dev                                                                                    
  git push origin main                                                                             
  2. Create feature branch for next work:                                                          
  git checkout dev                                                                                 
  git checkout -b feature/prompt-improvement                                                       
  3. Develop in this branch, then create a PR back to dev when done                                
                                                                                                   
  Why this is better:                                                                              
  - ✅ Keeps dev stable for integration                                                            
  - ✅ Each feature isolated in its own branch                                                     
  - ✅ Easy to revert if something breaks                                                          
  - ✅ Industry standard practice                                                                  
  - ✅ Easy collaboration (others can work on different features simultaneously)                   
                                                                                                   
  Your next work (prompt, chunking, tables):                                                       
  # For each task, create a separate branch                                                        
  git checkout -b feature/prompt-not-enough-info                                                   
  git checkout -b feature/chunking-overlap-50                                                      
  git checkout -b feature/table-handling                                                           
                                                                                                   
  ---                                                                                              
  Quick Command Cheat Sheet:                                                                       
                                                                                                   
  # See all branches                                                                               
  git branch -a                                                                                    
                                                                                                   
  # Create and switch to new branch                                                                
  git checkout -b feature/my-feature                                                               
                                                                                                   
  # Push new branch                                                                                
  git push origin feature/my-feature                                                               
                                                                                                   
  # Switch to existing branch                                                                      
  git checkout dev                                                                                 
                                                                                                   
  # Merge branch into current branch                                                               
  git merge feature/my-feature                                                                     
                                                                                                   
  # Delete branch (after merged)                                                                   
  git branch -d feature/my-feature                                                                 
                                                                                                   
  ---                                   