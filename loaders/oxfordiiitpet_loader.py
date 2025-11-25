from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image


def load_oxford_iiit_pet(
    root,
    download: bool = True,
):
    # define transforms
    image_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=Image.NEAREST),  # type: ignore
        ]
    )

    train_dataset = OxfordIIITPet(
        root=root,
        split="trainval",
        transform=image_transform,
        target_transform=mask_transform,
        download=download,
    )

    test_dataset = OxfordIIITPet(
        root=root,
        split="test",
        transform=image_transform,
        target_transform=mask_transform,
        download=download,
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
