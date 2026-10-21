from open_mythos import (
    OpenMythos,
    mythos_1b,
)

cfg = mythos_1b()
model = OpenMythos(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
