class Object1D:
    def __init__(self, start: float, end: float, eps_r: float, mu_r: float, sigma_e: float, sigma_m: float, name: str, color: str) -> None:
        self.start = start
        self.end = end
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma_e = sigma_e
        self.sigma_m = sigma_m
        self.name = name
        self.color = color

    def __str__(self) -> str:
        return f"Object: {self.name} start: {self.start} end: {self.end} eps_r: {self.eps_r} mu_r: {self.mu_r} sigma_e: {self.sigma_e} sigma_m: {self.sigma_m}"
