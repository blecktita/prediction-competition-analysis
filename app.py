import streamlit as st
import requests
import docker
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# Test Scenarios Data Structure
@dataclass
class TestScenario:
    """Test scenario container with description and expected behavior."""
    name: str
    description: str
    data: Dict
    expected_behavior: str

class TestCases:
    """Collection of test scenarios for the prediction service."""
    
    @staticmethod
    def get_test_scenarios() -> List[TestScenario]:
        """Return list of test scenarios."""
        return [
            TestScenario(
                name="Large Electronics Category",
                description="High volume electronics category with strong ratings",
                data={
                    "item_count": 5000,
                    "vol_purchase_total": 2500000.0,
                    "high_rating_perc": 0.75,
                    "rating_mean": 4.2,
                    "rating_std": 0.8,
                    "best_seller_count": 45
                },
                expected_behavior="High probability of Amazon presence due to large market size and good ratings"
            ),
            TestScenario(
                name="Niche Craft Category",
                description="Small, specialized craft supplies category",
                data={
                    "item_count": 150,
                    "vol_purchase_total": 25000.0,
                    "high_rating_perc": 0.82,
                    "rating_mean": 4.5,
                    "rating_std": 0.4,
                    "best_seller_count": 3
                },
                expected_behavior="Lower probability of Amazon presence due to small market size"
            ),
            # Add more scenarios from your test file here
        ]

def get_docker_client():
    """Get Docker client with error handling."""
    try:
        return docker.from_env()
    except Exception as e:
        st.error(f"Failed to connect to Docker: {e}")
        st.error("Please make sure Docker is running on your system.")
        return None

def check_docker_container(image_name: str) -> bool:
    """Check if the Docker container is running and healthy."""
    client = get_docker_client()
    if not client:
        return False
    
    try:
        containers = client.containers.list(
            filters={
                'ancestor': image_name,
                'status': 'running'
            }
        )
        return len(containers) > 0
    except Exception as e:
        st.error(f"Error checking container status: {e}")
        return False

def start_docker_container(image_name: str) -> bool:
    """Start the Docker container with simplified configuration."""
    client = docker.from_env()
    
    try:
        # Pull the latest image
        st.info("Pulling latest image...")
        client.images.pull(image_name)
        
        # List all images to verify pull
        images = client.images.list()
        st.info(f"Available images: {[img.tags for img in images]}")
        
        # Remove any existing containers
        containers = client.containers.list(
            all=True,
            filters={'ancestor': image_name}
        )
        
        for container in containers:
            st.info(f"Removing old container: {container.id[:12]}")
            container.remove(force=True)
            
        # Start new container with simple configuration
        st.info("Starting container...")
        container = client.containers.run(
            image_name,
            detach=True,
            ports={9696: 9696},  # Simplified port mapping
            remove=False  # Keep container around for debugging
        )
        
        # Wait for container to start
        time.sleep(5)
        
        try:
            # Get container logs
            logs = container.logs().decode('utf-8')
            st.code(f"Container logs:\n{logs}", language="bash")
            
            # Get container status
            container.reload()
            st.info(f"Container status: {container.status}")
            
            return container.status == 'running'
            
        except Exception as e:
            st.error(f"Error checking container status: {e}")
            return False
            
    except Exception as e:
        st.error(f"Error starting container: {e}")
        return False

def check_service_health(retries: int = 3) -> bool:
    """Check if the prediction service is responding."""
    url = 'http://localhost:9696/predict'
    for i in range(retries):
        try:
            response = requests.post(
                url,
                json={"item_count": 1},  # minimal test data
                timeout=2
            )
            if response.status_code == 400:  # Expected error for invalid data
                return True
            time.sleep(2)
        except requests.RequestException:
            if i < retries - 1:  # Don't show error unless it's the last try
                time.sleep(2)
                continue
            return False
    return False

def make_prediction(data: Dict) -> Optional[Dict]:
    """Make prediction using the API."""
    try:
        response = requests.post(
            'http://localhost:9696/predict',
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error making prediction: {e}")
        st.error("Please make sure the prediction service is running.")
        return None

def main():
    st.title("Amazon Category Predictor")
    st.write("Predict Amazon's presence in product categories")

    # Docker image configuration
    with st.sidebar:
        st.header("Service Configuration")
        docker_image = st.text_input(
            "Docker Image",
            "blecktita/amazon-competition-predictor:prod" 
        )

        # Service status indicator
        if check_docker_container(docker_image):
            st.success("🟢 Prediction service is running")
        else:
            st.error("🔴 Prediction service is not running")

        # Start container button
        if st.button("Start/Restart Prediction Service"):
            with st.spinner("Starting prediction service..."):
                if start_docker_container(docker_image):
                    if check_service_health():
                        st.success("Prediction service is ready!")
                    else:
                        st.error("Service started but not responding. Check container logs.")
                else:
                    st.error("Failed to start prediction service.")

    # Main area
    tab1, tab2 = st.tabs(["Custom Input", "Sample Categories"])
    
    with tab1:
        st.header("Category Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            item_count = st.number_input("Number of Items", min_value=0, value=1000)
            vol_purchase_total = st.number_input("Total Purchase Volume ($)", min_value=0.0, value=500000.0)
            high_rating_perc = st.slider("Percentage of High Ratings", 0.0, 1.0, 0.5)
        
        with col2:
            rating_mean = st.slider("Average Rating", 1.0, 5.0, 4.0)
            rating_std = st.slider("Rating Standard Deviation", 0.0, 2.0, 0.8)
            best_seller_count = st.number_input("Number of Best Sellers", min_value=0, value=10)

    with tab2:
        st.header("Sample Categories")
        scenarios = TestCases.get_test_scenarios()
        selected_scenario = st.selectbox(
            "Choose a pre-defined category",
            options=[s.name for s in scenarios]
        )

        # Get selected scenario
        scenario = next(s for s in scenarios if s.name == selected_scenario)
        
        # Display scenario details
        st.info(f"Description: {scenario.description}")
        st.info(f"Expected: {scenario.expected_behavior}")
        
        # Use scenario data
        item_count = scenario.data["item_count"]
        vol_purchase_total = scenario.data["vol_purchase_total"]
        high_rating_perc = scenario.data["high_rating_perc"]
        rating_mean = scenario.data["rating_mean"]
        rating_std = scenario.data["rating_std"]
        best_seller_count = scenario.data["best_seller_count"]

    # Make prediction
    if st.button("Make Prediction"):
        data = {
            "item_count": item_count,
            "vol_purchase_total": vol_purchase_total,
            "high_rating_perc": high_rating_perc,
            "rating_mean": rating_mean,
            "rating_std": rating_std,
            "best_seller_count": best_seller_count
        }

        with st.spinner("Making prediction..."):
            result = make_prediction(data)
            
            if result:
                st.header("Prediction Results")
                
                # Create columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display probability
                    probability = result["amazon_probability"]
                    st.metric(
                        "Amazon Presence Probability",
                        f"{probability:.1%}"
                    )
                
                with col2:
                    # Display prediction
                    if result["amazon_guess"]:
                        st.success("Prediction: Amazon is likely to be present in this category")
                    else:
                        st.info("Prediction: This category is likely to be Amazon-free")

if __name__ == "__main__":
    main()