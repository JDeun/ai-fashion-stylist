class UserInput:
    def __init__(self, gender, height, weight, budget, tpo, situation, image_paths):
        self.gender = gender
        self.height = height
        self.weight = weight
        self.budget = budget
        self.tpo = tpo
        self.situation = situation
        self.image_paths = image_paths

    @staticmethod
    def from_console():
        gender = input("성별을 입력하세요 (남성/여성): ")
        height = float(input("키를 입력하세요 (cm): "))
        weight = float(input("체중을 입력하세요 (kg): "))
        budget = int(input("가용 가능한 예산을 입력하세요 (원): "))
        tpo = input("코디가 필요한 상황(TPO)을 입력하세요: ")
        situation = input("구체적인 상황을 입력하세요: ")
        
        image_paths = []
        while True:
            path = input("분석할 이미지 파일 경로를 입력하세요 (입력 완료시 엔터): ")
            if path:
                image_paths.append(path)
            else:
                break

        return UserInput(gender, height, weight, budget, tpo, situation, image_paths)