let bst = null;
let currentNode = null;
let numberQueue = [];
let currentNumber = null;
let level = 0;
let positions = [];

// Generate list of random numbers and initialize game
function startGame() {
    numberQueue = Array.from({ length: 8 }, () => Math.floor(Math.random() * 100));
    bst = { value: numberQueue.shift(), left: null, right: null };
    currentNode = bst;
    level = 0;
    positions = [];
    displayTree();
    getNextNumber();
}

function getNextNumber() {
    if (numberQueue.length > 0) {
        currentNumber = numberQueue.shift();
        document.getElementById("next-number").textContent = `Next Number: ${currentNumber}`;
    } else {
        currentNumber = null;
        document.getElementById("next-number").textContent = "";
    }
}

function moveLeft() {
    if (!currentNode.left) {
        currentNode.left = { value: currentNumber, left: null, right: null };
        getNextNumber();
        resetToRoot();
    } else {
        currentNode = currentNode.left;
    }
    displayTree();
}

function moveRight() {
    if (!currentNode.right) {
        currentNode.right = { value: currentNumber, left: null, right: null };
        getNextNumber();
        resetToRoot();
    } else {
        currentNode = currentNode.right;
    }
    displayTree();
}

function changeLevel() {
    // Reset current node to root when changing level
    currentNode = bst;
    level = 0; // Reset level to root level
    positions = [];
    displayTree();
}

function resetToRoot() {
    currentNode = bst;
}

function finishGame() {
    if (validateBST(bst)) {
        document.getElementById("message").textContent = "Congratulations! You built the BST correctly!";
    } else {
        document.getElementById("message").textContent = "Incorrect BST. Try again!";
    }
}

function validateBST(node, min = null, max = null) {
    if (!node) return true;
    if ((min !== null && node.value <= min) || (max !== null && node.value >= max)) return false;
    return validateBST(node.left, min, node.value) && validateBST(node.right, node.value, max);
}

// Display BST - Visualization
function displayTree() {
    const treeContainer = document.getElementById("tree");
    treeContainer.innerHTML = ""; // Clear previous tree
    const treeWidth = 800;
    const nodeRadius = 40;
    
    // Render the tree starting from the root
    positions = [];
    renderNode(bst, treeWidth / 2, 50, treeWidth / 4, positions);

    // Now create the node elements based on the positions
    positions.forEach(({ value, x, y, parentX, parentY }) => {
        const nodeElement = document.createElement("div");
        nodeElement.className = "node";
        nodeElement.style.left = `${x - nodeRadius / 2}px`;
        nodeElement.style.top = `${y}px`;
        nodeElement.textContent = value;

        treeContainer.appendChild(nodeElement);

        // Draw lines connecting nodes
        if (parentX !== undefined && parentY !== undefined) {
            const line = document.createElement("div");
            line.className = "line";
            line.style.left = `${parentX}px`;
            line.style.top = `${parentY}px`;
            line.style.height = `${y - parentY}px`;
            line.style.width = `${Math.abs(x - parentX)}px`;
            line.style.transform = `translateX(${Math.min(x, parentX)}px)`;
            treeContainer.appendChild(line);
        }
    });
}

function renderNode(node, x, y, offset, positions) {
    if (!node) return;

    // Record the position of the current node
    positions.push({ value: node.value, x, y });

    // If this node has a left child, render the left child
    if (node.left) {
        const newX = x - offset;
        const newY = y + 100;
        positions.push({ value: node.left.value, x: newX, y: newY, parentX: x, parentY: y });
        renderNode(node.left, newX, newY, offset / 2, positions);
    }

    // If this node has a right child, render the right child
    if (node.right) {
        const newX = x + offset;
        const newY = y + 100;
        positions.push({ value: node.right.value, x: newX, y: newY, parentX: x, parentY: y });
        renderNode(node.right, newX, newY, offset / 2, positions);
    }
}

document.addEventListener("DOMContentLoaded", () => startGame());