{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "47046817",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "from torch import nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "531b61a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Net(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Net, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 32, 3, 1)\n",
    "        self.conv2 = nn.Conv2d(32, 64, 3, 1)\n",
    "        self.dropout1 = nn.Dropout(0.25)\n",
    "        self.dropout2 = nn.Dropout(0.5)\n",
    "        self.fc1 = nn.Linear(9216, 128)\n",
    "        self.fc2 = nn.Linear(128, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.conv1(x)\n",
    "        x = F.relu(x)\n",
    "        x = self.conv2(x)\n",
    "        x = F.relu(x)\n",
    "        x = F.max_pool2d(x, 2)\n",
    "        x = self.dropout1(x)\n",
    "        x = torch.flatten(x, 1)\n",
    "        x = self.fc1(x)\n",
    "        x = F.relu(x)\n",
    "        x = self.dropout2(x)\n",
    "        x = self.fc2(x)\n",
    "        output = F.log_softmax(x, dim=1)\n",
    "        return output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cdbb9271",
   "metadata": {},
   "outputs": [],
   "source": [
    "image_path = \"/home/deveshdatwani/closetx/assets/515859_623_41.jpeg\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d52c1323",
   "metadata": {},
   "outputs": [],
   "source": [
    "image = np.asarray(Image.open(image_path).convert(\"L\").resize((28,28)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "329da770",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaAAAAGdCAYAAABU0qcqAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIUpJREFUeJzt3Xts1fX9x/HXaWlPC/RCKb1BwYIIU6DLmHR4YTgaoNucCFm8/QHGQHTFDJnTdFFRt6QbJpvRMPxng5mIt0Qgkg2jKCVOwIESwtSO1o6L0JaLPaf0Tvv9/UHofuX++dD23ZbnIzkJPee8+v2cb7+nrx7OOe8TCoIgEAAAvSzGegEAgGsTBQQAMEEBAQBMUEAAABMUEADABAUEADBBAQEATFBAAAATg6wXcK6Ojg4dOXJESUlJCoVC1ssBADgKgkD19fXKyclRTMzFH+f0uQI6cuSIcnNzrZcBALhKhw4d0qhRoy56eZ8roKSkJElnFp6cnGy8GgCAq2g0qtzc3M7f5xfTYwW0atUqvfDCC6qurlZ+fr5efvllTZs27bK5s//tlpycTAEBQD92uadReuRFCG+++aaWL1+uFStW6LPPPlN+fr7mzJmj2trantgcAKAf6pEC+uMf/6jFixfrwQcf1I033qhXXnlFgwcP1l//+tee2BwAoB/q9gJqbW3V7t27VVhY+L+NxMSosLBQ27dvP+/6LS0tikajXU4AgIGv2wvo+PHjam9vV2ZmZpfzMzMzVV1dfd71S0tLlZKS0nniFXAAcG0wfyNqSUmJIpFI5+nQoUPWSwIA9IJufxVcenq6YmNjVVNT0+X8mpoaZWVlnXf9cDiscDjc3csAAPRx3f4IKD4+XlOnTtWWLVs6z+vo6NCWLVs0ffr07t4cAKCf6pH3AS1fvlwLFy7U97//fU2bNk0vvviiGhoa9OCDD/bE5gAA/VCPFNA999yjY8eO6ZlnnlF1dbW++93vavPmzee9MAEAcO0KBUEQWC/i/4tGo0pJSVEkEmESArx9++23XrmWlhbnTGJionNm6NChzpm2tjbnzKBBfn9j+uYA6cp/j5u/Cg4AcG2igAAAJiggAIAJCggAYIICAgCYoIAAACYoIACACQoIAGCCAgIAmKCAAAAmKCAAgAkKCABggomD6PPO/XDDK1FcXOy1rePHjztnLvRBi5eTkZHhnAmFQs6ZkSNHOmckaeHChc4Zpt3DFY+AAAAmKCAAgAkKCABgggICAJiggAAAJiggAIAJCggAYIICAgCYoIAAACYoIACACQoIAGCCAgIAmKCAAAAmmIYNbz5Tqv/zn/84ZyKRiHOmo6PDOSNJX331lXPmm2++cc4EQeCcSUhIcM5MnDjROSNJJ0+edM785Cc/cc5MmjTJOTNs2DDnDPomHgEBAExQQAAAExQQAMAEBQQAMEEBAQBMUEAAABMUEADABAUEADBBAQEATFBAAAATFBAAwAQFBAAwwTDSAebw4cPOmc8++8xrWwcOHHDOtLa2OmdGjBjhnJk+fbpzRpI2bdrknPEZyhobG+ucGTVqlHMmOTnZOSNJX3/9tXPm5Zdfds6MHDnSOTNhwgTnzMyZM50zkv8wV1wZHgEBAExQQAAAExQQAMAEBQQAMEEBAQBMUEAAABMUEADABAUEADBBAQEATFBAAAATFBAAwAQFBAAwwTDSPuzLL790zvzjH//ogZVc2KBB7odPEATOmZMnTzpnfIZcStLPfvYz58yePXucM9nZ2c4Zn6GsbW1tzhlfPj/bpqYm58zOnTudM59++qlzRpLmz5/vnPnpT3/qta1rEY+AAAAmKCAAgIluL6Bnn31WoVCoy4nP1AAAnKtHngO66aab9MEHH/xvIx7PFQAABrYeaYZBgwYpKyurJ741AGCA6JHngPbv36+cnByNHTtWDzzwgA4ePHjR67a0tCgajXY5AQAGvm4voIKCAq1du1abN2/W6tWrVVVVpdtvv1319fUXvH5paalSUlI6T7m5ud29JABAH9TtBVRUVKSf//znmjJliubMmaO///3vqqur01tvvXXB65eUlCgSiXSeDh061N1LAgD0QT3+6oDU1FTdcMMNqqiouODl4XBY4XC4p5cBAOhjevx9QKdOnVJlZaXXO78BAANXtxfQ448/rrKyMv33v//VJ598orvvvluxsbG67777untTAIB+rNv/C+7w4cO67777dOLECY0YMUK33XabduzY4TXHCgAwcHV7Ab3xxhvd/S0HhFOnTjln3nvvPedMJBJxzgwfPtw546u5ublXtlNbW+uV27dvn3Nm2LBhzhmfwZ1ffPGFcyYvL885I0kJCQnOmbi4OOeMz36Ij493zsTGxjpnJGnt2rXOmRtvvNE5M3bsWOfMQMAsOACACQoI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D+Xl5Tp48OCAOh4utx8uZM+ePZLUt44H61dBXIk33ngjCIfDwdq1a4MvvvgiWLJkSZCamhpUV1dbL61X/epXvwq2bt0aVFVVBf/85z+DwsLCID09PaitrbVeWo+qr68PPv/88+Dzzz8PJAV//OMfg88//zw4cOBAEARB8Pvf/z5ITU0NNm7cGOzduze46667gry8vKCpqcl45d3rUvuhvr4+ePzxx4Pt27cHVVVVwQcffBB873vfC8aPHx80NzdbL73bPPLII0FKSkqwdevW4OjRo52nxsbGzus8/PDDwejRo4MPP/ww2LVrVzB9+vRg+vTphqvufpfbDxUVFcHzzz8f7Nq1K6iqqgo2btwYjB07NpgxY4bxyrvqFwUUBEHw8ssvB6NHjw7i4+ODadOmBTt27LBeUq+75557guzs7CA+Pj4YOXJkcM899wQVFRXWy+pxH330USDpvNPChQuDIDjzUuynn346yMzMDMLhcDBr1qygvLzcdtE94FL7obGxMZg9e3YwYsSIIC4uLhgzZkywePHiAfdH2oVuv6RgzZo1nddpamoKfvGLXwTDhg0LBg8eHNx9993B0aNH7RbdAy63Hw4ePBjMmDEjSEtLC8LhcHD99dcHv/71r4NIJGK78HPwcQwAABN9/jkgAMDARAEBAExQQAAAExQQAMAEBQQAMEEBAQBMUEAAABMUEADABAUEADBBAQEATFBAAAATFBAAwMT/AaDKWT8r2bvrAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(image, cmap='gray')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d4b858ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_473421/2218935766.py:2: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.\n",
      "  model.load_state_dict(torch.load(\"/home/deveshdatwani/closetx/mnist/mnist_cnn.pt\"))\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = Net()\n",
    "model.load_state_dict(torch.load(\"/home/deveshdatwani/closetx/mnist/mnist_cnn.pt\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cf2171f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_473421/1669578956.py:1: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)\n",
      "  image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)\n"
     ]
    }
   ],
   "source": [
    "image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8c8f08b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([1, 1, 28, 28])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "image_tensor.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "fffebb24",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[ -566.8282,  -658.8439,     0.0000,  -377.6093,  -815.0148,  -119.4655,\n",
      "          -623.3838,  -786.0997,  -327.9395, -1198.5066]],\n",
      "       grad_fn=<LogSoftmaxBackward0>)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "tensor(3)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(model(image_tensor.float()))\n",
    "np.argmax(model(image_tensor.float()).detach())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "05695aae",
   "metadata": {},
   "outputs": [],
   "source": [
    "rand_input = torch.rand((1,28,28))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d5d3a762",
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "mat1 and mat2 shapes cannot be multiplied (64x144 and 9216x128)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[11], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[43mmodel\u001b[49m\u001b[43m(\u001b[49m\u001b[43mrand_input\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/lab/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1736\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1734\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[1;32m   1735\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m-> 1736\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/lab/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1747\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1742\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[1;32m   1743\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[1;32m   1744\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[1;32m   1745\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[1;32m   1746\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[0;32m-> 1747\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1749\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m   1750\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "Cell \u001b[0;32mIn[2], line 19\u001b[0m, in \u001b[0;36mNet.forward\u001b[0;34m(self, x)\u001b[0m\n\u001b[1;32m     17\u001b[0m x \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdropout1(x)\n\u001b[1;32m     18\u001b[0m x \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mflatten(x, \u001b[38;5;241m1\u001b[39m)\n\u001b[0;32m---> 19\u001b[0m x \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfc1\u001b[49m\u001b[43m(\u001b[49m\u001b[43mx\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     20\u001b[0m x \u001b[38;5;241m=\u001b[39m F\u001b[38;5;241m.\u001b[39mrelu(x)\n\u001b[1;32m     21\u001b[0m x \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdropout2(x)\n",
      "File \u001b[0;32m~/lab/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1736\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1734\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[1;32m   1735\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m-> 1736\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/lab/venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1747\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1742\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[1;32m   1743\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[1;32m   1744\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[1;32m   1745\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[1;32m   1746\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[0;32m-> 1747\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1749\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m   1750\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "File \u001b[0;32m~/lab/venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:125\u001b[0m, in \u001b[0;36mLinear.forward\u001b[0;34m(self, input)\u001b[0m\n\u001b[1;32m    124\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mforward\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;28minput\u001b[39m: Tensor) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Tensor:\n\u001b[0;32m--> 125\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mF\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mlinear\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43minput\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mweight\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbias\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[0;31mRuntimeError\u001b[0m: mat1 and mat2 shapes cannot be multiplied (64x144 and 9216x128)"
     ]
    }
   ],
   "source": [
    "model(rand_input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea9f3d28",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "my-virtualenv-name",
   "language": "python",
   "name": "my-virtualenv-name"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
