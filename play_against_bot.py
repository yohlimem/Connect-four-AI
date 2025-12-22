import uvicorn
import os

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the path to the web directory
    web_dir = os.path.join(script_dir, "web")
    
    # Change the current working directory to the script's directory
    # This makes sure that the relative paths in web/main.py work correctly
    os.chdir(script_dir)
    
    print("Starting web server...")
    print("Open http://localhost:8000 in your browser to play.")
    
    # Run the uvicorn server
    # We specify the app as 'main:app' and the directory as 'web'
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=".")