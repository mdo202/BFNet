import MtoBF
import IMGtoM
import pandas as pd

if __name__ == "__main__":
    
    '''
    Note: If replicating, you must manually select a task. You can't 
    iterate it through a list because your GPU memory will run out.
    '''
    
    # tasks = ["Neck", "Chest", "Abdomen", "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"]
    # task = "Wrist" 
    # # Completed: Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist
    
    # IMGtoM.train(task, save=False)

    # MtoBF.train(save=False)
    
    '''
    TESTING INSTRUCTIONS:
    
        1. Load a front and side image of a person hitting an 'A' pose (arms down diagonally)
           with camera positioning at 2.4 m (7.874 ft) away and 1.6 m (5.25 ft) high.
           
            Put said images in the 'Test' directory, and name the respective images 'test_front' 
            and 'test_side'.
            
        2. Input 
        
                Sex (M/F), Age, Weight (lbs), Height (ft- e.g. 5'9)
                
            in said order!!!!
    
    '''
    
    # Get user input for personal data
    sex_input = input("Enter your sex (M for Male, F for Female): ").strip().upper()
    age_input = int(input("Enter your age (in years): ").strip())
    weight_input = float(input("Enter your weight (in pounds): ").strip())
    height_input = input("Enter your height (e.g., 5'9 for 5 feet 9 inches): ").strip()

    # Convert height to meters
    feet, inches = map(int, height_input.split("'"))
    height_meters = (feet * 12 + inches) * 0.0254

    # Map Sex to numeric value
    sex_numeric = 1 if sex_input == 'M' else 0

    # Prepare the input DataFrame for MtoBF
    measurement_predictions = IMGtoM.test()

    # Create a DataFrame with all the inputs
    test_data = pd.DataFrame({
        'Sex': [sex_numeric],
        'Age': [age_input],
        'Weight': [weight_input * 0.453592],  # Convert weight to kilograms
        'Height': [height_meters],
        'Neck': [measurement_predictions['Neck']],
        'Chest': [measurement_predictions['Chest']],
        'Abdomen': [measurement_predictions['Abdomen']],
        'Hip': [measurement_predictions['Hip']],
        'Thigh': [measurement_predictions['Thigh']],
        'Knee': [measurement_predictions['Knee']],
        'Ankle': [measurement_predictions['Ankle']],
        'Biceps': [measurement_predictions['Biceps']],
        'Forearm': [measurement_predictions['Forearm']],
        'Wrist': [measurement_predictions['Wrist']]
    })

    # Predict body fat percentage using MtoBF
    predicted_body_fat = MtoBF.test(test_data)
    print(f"Predicted Body Fat Percentage: {predicted_body_fat[0]:.2f}%")
    
    
    