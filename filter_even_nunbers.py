# 建立一個命名為 filter_even_numbers.py 檔案
# 傳入一組數列，回傳其中的偶數
def get_even_numbers(numbers):
    even_numbers = []
    for number in numbers:
        if number % 2 == 0:
            even_numbers.append(number)
    return even_numbers

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = get_even_numbers(numbers)
print(even_numbers)

# 計算攝氏溫度華氏溫度轉換
def convert_temperature(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

celsius = 25
fahrenheit = convert_temperature(celsius)
print(fahrenheit)

# 計算 BMI
def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    return bmi

weight = 71.2
height = 1.76
bmi = calculate_bmi(weight, height)
print(bmi)