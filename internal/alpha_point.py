import numpy as np

class Alpha_point_Class:
  def __init__(self, seed, discretization_granularity):
      """
      Parameters:
          seed (float): random seed
          discretization_granularity(int): rounding sacle

      The Core Trade-off:
          The entire purpose of α-point rounding in this paper is to make the memoization technique work for counter data.
          Memoization provides strong privacy if a user's value stays the same, but it fails if the value changes even slightly.
          Rounding is used to group similar values together.
          Large user_custom_scale (e.g., ucs = D/2): Prioritizes stronger privacy at the cost of utility.
          Small user_custom_scale (e.g., ucs = 2): Prioritizes higher utility/accuracy at the cost of privacy in repeated collections.
      """
      self.rng = np.random.default_rng(seed=seed)
      self.discretization_granularity = discretization_granularity

  def alpha_point_rounding(self, x, a):
      """
      Performs alpha-point rounding on a single number.

      Args:
        x: The input number.
        a: Each user gets their own random "custom scale" [0,s)

      Returns:
        The rounded number.
      """
      s = self.discretization_granularity

      L = np.floor(x / s) * s
      R = L + s

      if x + a < R:
        return L
      else:
        return R

  def alpha_point_dataset(self, evolution_dataset):
      print('alpha point dataset ...')
      rounded_evolution_dataset = []

      for user_row in evolution_dataset:
          user_custom_scale = self.rng.integers(0, self.discretization_granularity)
          rounded_evolution_dataset.append([self.alpha_point_rounding(x, user_custom_scale) for x in user_row])

      return rounded_evolution_dataset


if __name__ == "__main__":
    
  # --- Testing the implementation ---

  # Parameters
  discretization_granularity = 10  # This is 's'

  alpha_point_obj = Alpha_point_Class(10, discretization_granularity)

  test_numbers = np.array([19, 23, 8, 42, 99, 101, 75, 30])
  num_simulations = 1000  # Run multiple times to get a stable MSE

  print(f"Discretization Granularity (s): {discretization_granularity}\n")
  print(f"Original Numbers: {test_numbers}\n")


  # --- Single Run Example ---
  print("--- Single Run Example ---")
  rounded_single_run = [alpha_point_obj.alpha_point_rounding(num, discretization_granularity/2) for num in test_numbers]
  print(f"Rounded Numbers (one instance): {rounded_single_run}\n")


  # --- MSE Calculation over multiple simulations ---
  print("--- MSE Calculation ---")
  squared_errors = []

  for _ in range(num_simulations):
    # Each user gets their own random "custom scale"
    a = np.random.randint(0, discretization_granularity)
    rounded_numbers = [alpha_point_obj.alpha_point_rounding(num, a) for num in test_numbers]
    errors = rounded_numbers - test_numbers
    squared_errors.extend(errors**2)

  mean_squared_error = np.mean(squared_errors)

  print(f"Mean Squared Error (MSE) over {num_simulations} simulations: {mean_squared_error:.4f}")

  # --- Verifying the Unbiased Nature ---
  print("\n--- Verifying Unbiased Nature ---")
  average_rounded_values = np.zeros(len(test_numbers))
  for _ in range(num_simulations):
      a = np.random.randint(0, discretization_granularity)
      rounded_numbers = [alpha_point_obj.alpha_point_rounding(num, a) for num in test_numbers]
      average_rounded_values += rounded_numbers

  average_rounded_values /= num_simulations

  print(f"Original Averages: {np.mean(test_numbers):.4f}")
  print(f"Average of Rounded Values over simulations: {np.mean(average_rounded_values):.4f}")

  print("\nAverage rounded value for each original number:")
  for orig, avg_round in zip(test_numbers, average_rounded_values):
      print(f"  Original: {orig}, Average Rounded: {avg_round:.4f}")
