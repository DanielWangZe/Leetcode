import java.util.*;

class MyCircularQueue {
    final int[] a;
    int front = 0, rear = -1, len = 0;

    public MyCircularQueue(int k) { a = new int[k];}

    public boolean enQueue(int val) {
        if (!isFull()) {
            rear = (rear + 1) % a.length;
            a[rear] = val;
            len++;
            return true;
        } else return false;
    }

    public boolean deQueue() {
        if (!isEmpty()) {
            front = (front + 1) % a.length;
            len--;
            return true;
        } else return false;
    }

    public int Front() { return isEmpty() ? -1 : a[front];}

    public int Rear() {return isEmpty() ? -1 : a[rear];}

    public boolean isEmpty() { return len == 0;}

    public boolean isFull() { return len == a.length;}
}

class TreeNode{
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x;}
}

class ListNode{
    int val;
    ListNode next;
    ListNode(int x) { val = x;}
}

class LRUCache{

    class DLinkNode{
        int key;
        int value;
        DLinkNode prev;
        DLinkNode post;
    }


    private int count;
    private int capacity;
    private DLinkNode head, tail;
    private HashMap<Integer, DLinkNode> cache = new HashMap<>();

    private void removeNode(DLinkNode node){
        DLinkNode prev = node.prev;
        DLinkNode post = node.post;

        prev.post = post;
        post.prev = prev;
    }

    private void addNode(DLinkNode node){
        node.prev = head;
        node.post = head.post;

        head.post.prev = node;
        head.post = node;
    }

    private void moveToHead(DLinkNode node){
        this.removeNode(node);
        this.addNode(node);
    }

    private DLinkNode popTail(){
        DLinkNode res = tail.prev;
        this.removeNode(res);
        return res;
    }

    public LRUCache(int capacity){
        this.count = 0;
        this.capacity = capacity;

        head = new DLinkNode();
        head.prev = null;

        tail = new DLinkNode();
        tail.post = null;

        head.post = tail;
        tail.prev = head;
    }

    public int get(int key){
        DLinkNode node = cache.get(key);
        if(node == null) return -1;
        this.moveToHead(node);
        return node.value;
    }

    public void put(int key, int value){
        DLinkNode node = cache.get(key);
        if(node == null){
            DLinkNode newNode = new DLinkNode();
            newNode.key = key;
            newNode.value = value;

            this.cache.put(key, newNode);
            this.addNode(node);

            ++count;

            if(count > capacity){
                DLinkNode tail = this.popTail();
                this.cache.remove(tail.key);
                --count;
            }
        }else{
            node.value = value;
            this.moveToHead(node);
        }
    }
}

class MinStack{
    Deque<Integer> stack;
    Deque<Integer> min;

    public MinStack(){
        stack = new ArrayDeque<>();
        min = new ArrayDeque<>();
    }

    public void push(int x){
        stack.push(x);
        if(min.isEmpty() || x <= min.peek()) min.push(x);
    }

    public void pop(){
        int x = stack.pop();
        if(min.peek() == x) min.pop();
    }

    public int top(){
        return stack.peek();
    }

    public int getMin(){
        return min.peek();
    }
}

class Solution{
    public static int findTheFirstNoLessThan(int[] nums, int target){
        int left = 0, right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] < target){
                left = mid + 1;
            }
            else{
                right = mid - 1;
            }
        }
        return left;
    }

    public static int findTheFirstGreaterThan(int[] nums, int target){
        int left = 0, right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] <= target){ // the only difference
                left = mid + 1;
            }
            else{
                right = mid - 1;
            }
        }
        return left;
    }

    public static int findTheLastNoLessThan(int[] nums, int target){
        int left = 0, right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] <= target){ // the only difference
                left = mid + 1;
            }
            else{
                right = mid - 1;
            }
        }
        return left - 1;
    }

    public static int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while(left < right){
            int mid = left + (right - left) / 2;
            if(nums[mid] > nums[mid + 1]) left = mid + 1;
            else right = mid;
        }
        return right;
    }

    public List<Integer> preorderTraversalRecursive(TreeNode root){
        List<Integer> nodes = new ArrayList<Integer>();
        preorderTraversalHelper(root, nodes);
        return nodes;
    }
    private void preorderTraversalHelper(TreeNode root, List<Integer> l){
        if(root == null) return;
        else{
            l.add(root.val);
            preorderTraversalHelper(root.left, l);
            preorderTraversalHelper(root.right, l);
        }
    }

    public List<Integer> preorderTraversalIterative(TreeNode root){
        List<Integer> result = new ArrayList<>();
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode p = root;
        while(!stack.isEmpty() || p != null) {
            if(p != null) {
                stack.push(p);
                result.add(p.val);  // Add before going to children
                p = p.left;
            } else {
                TreeNode node = stack.pop();
                p = node.right;
            }
        }
        return result;
    }

    public List<Integer> inorderTraversalIterative(TreeNode root){
        List<Integer> result = new ArrayList<>();
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode p = root;
        while(!stack.isEmpty() || p != null) {
            if(p != null) {
                stack.push(p);
                p = p.left;
            } else {
                TreeNode node = stack.pop();
                result.add(node.val);  // Add after all left children
                p = node.right;
            }
        }
        return result;
    }

    public List<Integer> postorderTraversalIterative(TreeNode root){
        LinkedList<Integer> result = new LinkedList<>();
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode p = root;
        while(!stack.isEmpty() || p != null) {
            if(p != null) {
                stack.push(p);
                result.addFirst(p.val);  // Reverse the process of preorder
                p = p.right;             // Reverse the process of preorder
            } else {
                TreeNode node = stack.pop();
                p = node.left;           // Reverse the process of preorder
            }
        }
        return result;
    }

    // leecode 323.Number of Connected Components in an Undirected Graph
    // adjacency list + DFS
    public int countComponent(int n, int[][] edges) {
        Boolean[] visited = new Boolean[n];
        List<ArrayList<Integer>> list = new ArrayList<>(n);
        // build adjacency list first
        for (int[] edge: edges) {
            int from = edge[0];
            int to = edge[1];
            list.get(from).add(to);
            list.get(to).add(from);
        }

        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                ++res;
                dfs(list, visited, i);
            }
        }
        return res;
    }

    public void dfs(List<ArrayList<Integer>> l, Boolean[] v, Integer i) {
        if (v[i]) return;
        v[i] = true;
        for (int j = 0; j < l.get(i).size(); ++j) {
            dfs(l, v, l.get(i).get(j));
        }
    }

    public int maxDepth_topDown(TreeNode root){
        //pre order - top down
        int d = 0;
        maxDepthHelper(root, d);
        return d;
    }
    private int ans;
    private void maxDepthHelper(TreeNode root, int depth){
        if(root == null) return;
        if(root.left == null && root.right == null) ans = Math.max(ans, depth);
        maxDepthHelper(root.left, depth + 1);
        maxDepthHelper(root.right, depth + 1);
    }

    public int maxDepth_bottomUp(TreeNode root){
        if(root == null) return 0;
        int left = maxDepth_bottomUp(root.left);
        int right = maxDepth_bottomUp(root.right);
        return Math.max(left, right) + 1;
    }

    public boolean isSymmetric(TreeNode root){
        if(root == null) return true;
        return isSymmetricHelper(root.left, root.right);
    }

    private boolean isSymmetricHelper(TreeNode left, TreeNode right){
        if(left == null && right == null) return true;
        if(left == null || right == null) return false;
        if(left.val == right.val){
            return isSymmetricHelper(left.left, right.right) && isSymmetricHelper(left.right, right.left);
        }
        else{
            return false;
        }
    }

    public int[] productExceptSelf(int[] nums){
        int length = nums.length;
        int[] L = new int[length];
        int[] R = new int[length];
        int[] ret = new int[length];

        //initialize L, which is the product of all elements to the left of num[i]
        L[0] = 1;
        for(int i = 1; i < length; i++){
            L[i] = nums[i - 1] * L[i - 1];
        }
        R[length - 1] = 1;
        //initialize R, which is the product of all elements to the right of nums[i]
        for(int i = length - 2; i >= 0; i--){
            R[i] = R[i + 1] * nums[i + 1];
        }
        //initialize ret, which is the product of L and R element wise
        for(int i = 0; i < length; i++) {
            ret[i] = L[i] * R[i];
        }
        return ret;
    }

    public boolean isPowerOfFour(int num){
        if(num == 1) return true;
        if(num >= 4 && num % 4 == 0) return isPowerOfFour(num / 4);
        return false;
    }

    public boolean hasPathSum(TreeNode root, int sum){
        if(root == null) return false;
        if(root.val == sum && root.left == null && root.right == null) return true;
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    public void reverseString(char[] s){
        reverseStringHelper(0, s);
    }

    private void reverseStringHelper(int index, char[] s){
        if(s == null || index >= s.length){
            return;
        }
        reverseStringHelper(index + 1, s);
        System.out.print(s[index]);
    }

    public ListNode swapPairs_Recursion(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode n = head.next;
        head.next = swapPairs_Recursion(head.next.next);
        n.next = head;
        return n;
    }

    public ListNode swapPairs_Iterative(ListNode head){
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode current = dummy;
        while(current.next != null && current.next.next != null){
            ListNode first = current.next;
            ListNode second = current.next.next;
            first.next = second.next;
            current.next = second;
            current.next.next = first;
            current = current.next.next;
        }
        return dummy.next;
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> triangle = new ArrayList<List<Integer>>();
        if (numRows <=0){
            return triangle;
        }
        for (int i=0; i<numRows; i++){
            List<Integer> row =  new ArrayList<Integer>();
            for (int j=0; j<i+1; j++){
                if (j==0 || j==i){
                    row.add(1);
                } else {
                    row.add(triangle.get(i-1).get(j-1)+triangle.get(i-1).get(j));
                }
            }
            triangle.add(row);
        }
        return triangle;
    }

    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<Integer>();
        if(rowIndex < 0) return row;
        for(int i = 0; i < rowIndex + 1; i++){
            row.add(0, 1);
            for(int j = 1; j < row.size() - 1; j++){
                row.set(j, row.get(j) + row.get(j + 1));
            }
        }
        return row;
    }

    // leetcode 206 reverse linked list - recursion
    // p never change since the first return, head keep changing
    public ListNode reverseList_Recursion(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode p = reverseList_Recursion(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    // leetcode 206 reverse linked list - iteratively
    public ListNode reverseList_Iterative(ListNode head){
        ListNode curr = head;
        ListNode prev = null;
        ListNode next = null;
        while(curr != null){
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    public int fib(int N){
        HashMap<Integer, Integer> map = new HashMap<>();
        if(map.containsKey(N)) return map.get(N);
        if(N == 0 || N == 1){
            map.put(N, N);
            return N;
        }
        else{
            int ret = fib(N - 1) + fib(N - 2);
            map.put(N, ret);
            return ret;
        }
    }

    HashMap<Integer, Integer> stairMap = new HashMap<>();
    public int climbStairs(int n){
        if(stairMap.containsKey(n)) return stairMap.get(n);
        if(n == 0 || n == 1){
            stairMap.put(n, 1);
            return 1;
        }
        else{
            int ret = climbStairs(n - 1) + climbStairs(n - 2);
            stairMap.put(n, ret);
            return ret;
        }
    }

    public double myPow(double x, int n){
        double res = 1.0;
        for (int i = n; i != 0; i /= 2) {
            if (i % 2 != 0) res *= x;
            x *= x;
        }
        return n < 0 ? 1 / res : res;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2){
        ListNode dummy = new ListNode(-1), curr = dummy;
        while(l1 != null && l2 != null){
            if(l1.val < l2.val){
                curr.next = l1;
                l1 = l1.next;
            }else{
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }
        curr.next = l1 != null ? l1 : l2;
        return dummy.next;
    }

    public int kthGrammar(int N, int K){
        if(N == 1) return 0;
        if(K % 2 == 0) {
            return kthGrammar(N - 1, K / 2) == 0 ? 1 : 0;
        }else{
            return kthGrammar(N - 1, (K + 1) / 2) == 0 ? 0 : 1;
        }
    }

    // leetcode 11 container with most water
    // two pointer
    public int maxArea(int[] height){
        int res = 0, i = 0, j = height.length - 1;
        while(i < j){
            res = Math.max(res, Math.min(height[i], height[j]) * (j - i));
            if(height[i] < height[j]) i++;
            else j--;
        }
        return res;
    }

    public int maxProduct(int[] nums){
        if(nums == null) return 0;
        int curr_max_product = nums[0];
        int prev_max_product = nums[0];
        int curr_min_product = nums[0];
        int prev_min_product = nums[0];
        int ans = nums[0];
        for(int i = 1; i < nums.length; i++){
            curr_max_product = Math.max(Math.max(prev_max_product * nums[i], prev_min_product * nums[i]), nums[i]);
            curr_min_product = Math.min(Math.min(prev_max_product * nums[i], prev_min_product * nums[i]), nums[i]);
            ans = Math.max(ans, curr_max_product);
            prev_max_product = curr_max_product;
            prev_min_product = curr_min_product;
        }
        return ans;
    }

    // leetcode 560 subarray sum equals K
    // pre sum -> idea is to get sum[0, i-1] and sum[0, j] so we can get sum[i, j]
    // it is for those (sum - k) == 0 calculations which are valid subarrays but need to get counted.
    // e.g. if k = 7 and sum = 7 (at second element for array is : 3, 4, 3, 8) at some iteration.....
    // then sum - k = 0....this 0 will get counted in statement result += preSum.get(sum - k)
    public int subarraySum(int[] nums, int k){
        int sum = 0, result = 0;
        Map<Integer, Integer> preSum = new HashMap<>();
        preSum.put(0, 1);

        for(int i = 0; i < nums.length; i++){
            sum += nums[i];
            result += preSum.getOrDefault(sum - k, 0);
            preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);
        }
        return result;
    }

    public boolean canPartition(int[] nums){
        if(nums == null || nums.length == 0) return true;
        int target = 0;
        for(int num: nums){
            target += num;
        }
        if(target % 2 != 0) return false;
        target /= 2;

        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for(int i = 0; i < nums.length; i++){
            for(int j = target; j >= nums[i]; j--){
                dp[j] = dp[j] || dp[j - nums[i]];
            }
        }
        return dp[target];
    }

    private int lo, maxLen;
    public String longestPalindrome(String s){
        for(int i = 0; i < s.length(); i++){
            longestPalindromeExtend(s, i, i);
            longestPalindromeExtend(s, i, i+1);
        }
        return s.substring(lo, lo + maxLen);
    }

    private void longestPalindromeExtend(String s, int j, int k){
        while(j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)){
            j--;
            k++;
        }
        if(maxLen < k - j - 1){
            lo = j + 1;
            maxLen = k - j -1;
        }
    }

    public boolean canJump(int[] nums){
        int[] dp = new int[nums.length];
        for(int i = 1; i < nums.length; i++){
            dp[i] = Math.max(dp[i - 1], nums[i - 1]) - 1;
            if(dp[i] < 0) return false;
        }
        return true;
    }

    public List<List<Integer>> levelOrder(TreeNode root){
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            List<Integer> level = new ArrayList<>();
            int cnt = queue.size();
            for(int i = 0; i < cnt; i++){
                TreeNode node = queue.poll();
                level.add(node.val);
                if(node.left != null){
                    queue.add(node.left);
                }
                if(node.right != null){
                    queue.add(node.right);
                }
            }
            res.add(level);
        }
        return res;
    }

    // leetcode 200 Numbers of islands
    // DFS
    public int numIslands(char[][] grid){
        if (grid == null || grid.length == 0) return 0;
        int y = grid.length;
        int x = grid[0].length;
        int ans = 0;
        for(int i = 0; i < y; i++){
            for(int j = 0; j < x; j++){
                if(grid[i][j] == '1'){
                    ++ans;
                    dfs(i, j, y, x, grid);
                }
            }
        }
        return ans;
    }

    // leetcode 547 Friend Circles
    // DFS
    public int findCircleNum(int[][] M) {
        return 0;
    }

    private void dfs(int i, int j, int y, int x, char[][] grid){
        if(i < 0 || i >= y || j < 0 || j >= x || grid[i][j] == '0')
            return;
        grid[i][j] = '0';
        dfs(i - 1, j, y, x, grid);
        dfs(i + 1, j, y, x, grid);
        dfs(i, j - 1, y, x, grid);
        dfs(i, j + 1, y, x, grid);
    }

    public int openLock(String[] deadends, String target){
        String start = "0000";

        Set<String> dead = new HashSet();
        for(String d: deadends) dead.add(d);

        if(dead.contains(start)) return -1;

        Queue<String> queue = new LinkedList();
        queue.offer(start);

        Set<String> visited = new HashSet();
        visited.add(start);

        int steps = 0;
        while(!queue.isEmpty()){
            ++steps;
            int size = queue.size();
            for(int s = 0; s < size; ++s){
                String node = queue.poll();
                for(int i = 0; i < 4; ++i){
                    for(int j = -1; j <= 1; j+= 2){
                        char[] chars = node.toCharArray();
                        chars[i] = (char)(((chars[i] - '0') + j + 10) % 10 + '0');
                        String next = new String(chars);
                        if(next.equals(target)) return steps;
                        if(dead.contains(next) || visited.contains(next)) continue;
                        visited.add(next);
                        queue.offer(next);
                    }
                }
            }
        }
        return -1;
    }

    public int numSquares(int n){
        int dp[] = new int[n + 1];
        Arrays.fill(dp, n);
        dp[0] = 0;
        dp[1] = 1;
        for(int i = 1; i <= n; i++){
            for(int j = 0; j * j <= i; j++){
                dp[i] = Math.min(dp[i], 1 + dp[i - j * j]);
            }
        }
        return dp[n];
    }

    // leetcode 20 Valid Parentheses
    // stack - peek vs pop
    public boolean isValid(String s){
        char[] chars = s.toCharArray();
        if(chars.length % 2 != 0) return false;
        Stack<Character> stack = new Stack<>();
        for(char c: chars){
            if(c == '(' || c == '{' || c == '['){
                stack.push(c);
            } else{
                if(stack.isEmpty()) return false;
                char top = stack.peek();
                if(isMatch(top, c)) {
                    stack.pop();
                    continue;
                } else{
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    private boolean isMatch(char left, char right){
        if(left == '(' && right == ')') return true;
        if(left == '{' && right == '}') return true;
        if(left == '[' && right == ']') return true;
        return false;
    }

    public int[] dailyTemperature(int[] T){
        int n = T.length;
        int[] res = new int[n];
        Arrays.fill(res, 0);
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < n; ++i){
            while(!stack.isEmpty() && T[i] > T[stack.peek()]){
                int top = stack.peek();
                stack.pop();
                res[top] = i - top;
            }
            stack.push(i);
        }
        return res;
    }

    public int evalRPN(String[] tokens){
        int n = tokens.length;
        Stack<String> s = new Stack<>();
        for(int i = 0; i < n; ++i){
            if(!tokens[i].equals("+") && !tokens[i].equals("-") && !tokens[i].equals("*") && !tokens[i].equals("/")){
                s.push(tokens[i]);
            }else {
                int right = Integer.parseInt(s.pop());
                int left = Integer.parseInt(s.pop());
                int res = 0;
                if (tokens[i].equals("+")) res = left + right;
                if (tokens[i].equals("-")) res = left - right;
                if (tokens[i].equals("*")) res = left * right;
                if (tokens[i].equals("/")) res = left / right;
                s.push(Integer.toString(res));
            }
        }
        return Integer.parseInt(s.pop());
    }

    class Node{
        public int val;
        public List<Node> neighbors;

        public Node() {}

        public Node(int _val, List<Node> _neighbors){
            val = _val;
            neighbors = _neighbors;
        }
    }

    public Node cloneGraph(Node node){
        Queue<Node> q = new LinkedList<>();
        Map<Node, Node> map = new HashMap<>();
        q.offer(node);
        map.put(node, new Node(node.val, new ArrayList<>()));
        while(!q.isEmpty()){
            Node n = q.poll();
            for(Node l: n.neighbors){
                // clone nodes
                if(!map.containsKey(l)){
                    map.put(l, new Node(l.val, new ArrayList<>()));
                    q.add(l);
                }
                // cloe edges
                map.get(n).neighbors.add(map.get(l));
            }
        }
        return map.get(node);
    }

    public int[][] updateMatrix(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    queue.offer(new int[] {i, j});
                }
                else {
                    matrix[i][j] = Integer.MAX_VALUE;
                }
            }
        }

        int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            for (int[] d : dirs) {
                int r = cell[0] + d[0];
                int c = cell[1] + d[1];
                if (r < 0 || r >= m || c < 0 || c >= n ||
                        matrix[r][c] <= matrix[cell[0]][cell[1]] + 1) continue;
                queue.add(new int[] {r, c});
                matrix[r][c] = matrix[cell[0]][cell[1]] + 1;
            }
        }

        return matrix;
    }

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor){
        if (image[sr][sc] == newColor) return image;
        int m = image.length;
        int n = image[0].length;

        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{sr, sc});

        int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int oldColor = image[sr][sc];

        while(!queue.isEmpty()){
            int[] point = queue.poll();
            image[point[0]][point[1]] = newColor;
            for(int[] d: dirs){
                int r = point[0] + d[0];
                int c = point[1] + d[1];
                if(r >= 0 && r < m && c >= 0 && c < n && image[r][c] == oldColor){
                    queue.offer(new int[]{r, c});
                }
            }
        }
        return image;
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms){
        int len = rooms.size();
        boolean[] record = new boolean[len];
        visitRoomsHelper(0, rooms, record);
        for(int i = 0; i < len; ++i){
            if(record[i]) continue;
            return false;
        }
        return true;
    }

    void visitRoomsHelper(int i, List<List<Integer>> rooms, boolean[] record){
        if(record[i]) return;
        record[i] = true;
        for(int r: rooms.get(i)){
            visitRoomsHelper(r, rooms, record);
        }
    }

    public int minEatingSpeed(int[] piles, int H){
        int left = 1;
        int right = (int)Math.pow(10.0, 9.0) + 1;
        while(left < right){
            int mid = left + (right - left) / 2;
            int t = 0;
            for(int p: piles){
                t += (p + mid - 1) / mid;
            }
            if(t > H) left = mid + 1;
            else right = mid;
        }
        return left;
    }

    public int[] intersect(int[] nums1, int[] nums2){
        Map<Integer, Integer> m = new HashMap<>();
        List<Integer> result = new ArrayList<Integer>();
        for(int n: nums1){
            m.put(n, m.getOrDefault(n, 0) + 1);
        }
        for(int n: nums2){
            if(m.containsKey(n) && m.get(n) > 0){
                result.add(n);
                m.put(n, m.get(n) - 1);
            }
        }
        return result.stream().mapToInt(i->i).toArray();
    }

    public int findRadius(int[] houses, int[] heaters){
        Arrays.sort(houses);
        Arrays.sort(heaters);

        int i = 0, j = 0, res = 0;
        while(i < houses.length){
            while(j < heaters.length - 1 &&
                    (Math.abs(heaters[j + 1] - houses[i]) <= Math.abs(heaters[j] - houses[i]))){
                j++;
            }
            res = Math.max(res, Math.abs(heaters[j] - houses[i]));
            i++;
        }
        return res;
    }

    public int minSubArrayLen(int s, int[] nums){
        int sum = 0, from = 0, len = Integer.MAX_VALUE;
        for(int i = 0; i < nums.length; i++) {
            sum += nums[i];
            while(sum >= s) {
                len = Math.min(len, i - from + 1);
                sum -= nums[from++];
            }
        }
        return len == Integer.MAX_VALUE ? 0 : len;
    }

    public boolean isPalindrome(String s) {
        if(s == null || s.length() == 0) return true;
        char[] ss = s.toCharArray();
        int left = 0, right = ss.length - 1;
        while(left < right) {
            while(left < right && isNeitherDigitNorLetter(ss[left])) ++left;
            while(left < right && isNeitherDigitNorLetter(ss[right])) --right;
            if(Character.toLowerCase(ss[left++]) != Character.toLowerCase(ss[right--])) return false;
        }
        return true;
    }

    private boolean isNeitherDigitNorLetter(char c) {
        return (!Character.isDigit(c)) && (!Character.isLetter(c));
    }

    public int longestOnes(int[] A, int K){
        int left = 0, res = 0, zeros = 0;
        for(int right = 0; right < A.length; ++right) {
            if (A[right] == 0) ++zeros;
            while (zeros > K) {
                if (A[left++] == 0) --zeros;
            }
            res = Math.max(right - left + 1, res);
        }
        return res;
    }

    public int[] sortedSquares(int[] A) {
        int len = A.length;
        int[] res = new int[len];
        int left = 0, right = len - 1, index = len - 1;
        while (left <= right) {
            if (Math.abs(A[left]) < Math.abs(A[right])) {
                res[index] = A[right] * A[right];
                right--;
            } else {
                res[index] = A[left] * A[left];
                left++;
            }
            index--;
        }
        return res;
    }

    public int bagOfTokensScore(int[] tokens, int P) {
        Arrays.sort(tokens);
        int i = 0, j = tokens.length - 1, ans = 0, points = 0;
        while(i <= j) {
            if(P >= tokens[i]) {
                P -= tokens[i++];
                points += 1;
                ans = Math.max(ans, points);
            }
            else if(points > 0) {
                P += tokens[j--];
                points -= 1;
            }
            else break;
        }
        return ans;
    }

    public boolean isLongPressedName(String name, String typed) {
        char[] n = name.toCharArray();
        char[] t = typed.toCharArray();
        int i = 0, j = 0;
        while(i < n.length && j < t.length) {
            if(n[i] == t[j]) {
                ++i;
                ++j;
            } else if (j > 0 && t[j] == t[j - 1]) {
                ++j;
            } else {
                return false;
            }
        }
        while(j < t.length && t[j] == t[j - 1]) ++j;
        return i == n.length && j == t.length;
    }

    public String reverseOnlyLetters(String S) {
        char[] s = S.toCharArray();
        int i = 0, j = s.length - 1;
        while(i <= j) {
            while(i <= j && !Character.isLetter(s[i]))  ++i;
            while(i <= j && !Character.isLetter(s[j]))  --j;
            if(i <= j) {
                char tmp = s[i];
                s[i] = s[j];
                s[j] = tmp;
                ++i;
                --j;
            }
        }
        return String.copyValueOf(s);
    }

    // leetcode 15 three sum
    // sort + two sum(two pointer)
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums == null || nums.length == 0) return res;
        Arrays.sort(nums);
        for(int i = 0; i < nums.length - 2; ++i) {
            if(nums[i] > 0) break;
            if(i > 0 && nums[i] == nums[i - 1]) continue; // not ++i, this is for handling duplicates
            int l = i + 1;
            int r = nums.length - 1;
            while(l < r) {
                if (nums[i] + nums[l] + nums[r] == 0) {
                    List<Integer> ans = new ArrayList<>();
                    ans.add(nums[i]);
                    ans.add(nums[l++]);
                    ans.add(nums[r--]);
                    res.add(ans);
                    while(l < r && nums[l] == nums[l - 1]) ++l; // handling duplicates
                    while(l < r && nums[r] == nums[r + 1]) --r; // handling duplicates
                } else if (nums[i] + nums[l] + nums[r] < 0) {
                    ++l;
                } else {
                    --r;
                }
            }
        }
        return res;
    }

    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode slow = head, fast = head.next;
        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode middle = slow.next;
        slow.next = null;
        return mergeSortedList(sortList(head), sortList(middle));
    }

    private ListNode mergeSortedList(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1), curr = dummy;
        while(l1 != null && l2 != null) {
            if(l1.val < l2.val) {
                curr.next = l1;
                l1 = l1.next;
            } else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }
        curr.next = l1 != null ? l1 : l2;
        return dummy.next;
    }

    public boolean isIdealPermutation(int[] A) {
        int n = A.length;
        for(int i = 0; i < n; ++i) {
            if(Math.abs(A[i] - i) > 1) return false;
        }
        return true;
    }

    public int majorityElement(int[] nums) {
        Map<Integer, Integer> m = new HashMap<>();
        for(int n : nums) {
            m.put(n, m.getOrDefault(n, 0) + 1);
            if(m.get(n) > nums.length / 2) return n;
        }
        return -1;
    }

    public int numPrimeArrangements(int n) {
        int numPrime = 0;
        int ans = 1;
        for(int i = 1; i <= n; ++i) {
            if(isPrime(i)) numPrime += 1;
        }
        for (int i = 1; i <= numPrime; ++i) // (# of primes)!
            ans = ans * i % 1_000_000_007;
        for (int i = 1; i <= n - numPrime; ++i) // (# of non-primes)!
            ans = ans * i % 1_000_000_007;
        return (int)ans;
    }

    private boolean isPrime(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        for(int i = 2; i <= Math.sqrt(n); ++i) {
            if(n % i == 0) return false;
        }
        return true;
    }

    public int dietPlanPerformance(int[] calories, int k, int lower, int upper) {
        int n = calories.length;
        int points = 0;
        int sum = 0;
        for(int i = 0; i < k - 1; ++i) {
            sum += calories[i];
        }
        for(int i = k - 1; i < n; ++i) {
            if(i >= k) sum -= calories[i - k];
            sum += calories[i];
            if(sum > upper) ++points;
            if(sum < lower) --points;
        }
        return points;
    }

    public int distanceBetweenBusStops(int[] distance, int start, int destination) {
        int totalSum = 0;
        int clockWiseSum = 0;
        int newStart = Math.min(start, destination);
        int newDest = Math.max(start, destination);
        for (int i = 0; i < distance.length; ++i) {
            totalSum += distance[i];
        }
        for (int j = newStart; j < newDest; ++j) {
            clockWiseSum += distance[j];
        }
        return Math.min(clockWiseSum, totalSum - clockWiseSum);
    }

    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[2];
        res[0] = firstPos(nums, target);
        res[1] = lastPos(nums, target);
        return res;
    }

    private int firstPos(int[] nums, int target) {
        int l = 0;
        int r = nums.length;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (nums[m] < target) {
               l = m + 1;
            } else {
                r = m;
            }
        }
        if (l == nums.length || nums[l] != target) return -1;
        return l;
    }

    private int lastPos(int[] nums, int target) {
        int l = 0;
        int r = nums.length;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (nums[m] <= target) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        --l;
        if (l < 0 || nums[l] != target) return -1;
        return l;
    }

    public int[] rearrangeBarcodes(int[] barcodes) {
        int[] arr = new int[10001];
        int maxF = 0, maxV = 0;
        for (int i = 0; i < barcodes.length; ++i) {
            if (++arr[barcodes[i]] > maxF) {
                maxF = arr[barcodes[i]];
                maxV = barcodes[i];
            }
        }

        int[] ans = new int[barcodes.length];
        int pos = 0;
        while (arr[maxV]-- > 0) {
            ans[pos] = maxV;
            if((pos += 2) >= barcodes.length) pos = 1;
        }
        for (int v = 1; v < 10001; ++v) {
            while (arr[v] -- > 0) {
                ans[pos] = v;
                if ((pos += 2) >= barcodes.length) pos = 1;
            }
        }
        return ans;
    }

    //leetcode 464 backtracking + memorization
    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if ((1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal) return false;
        int[] state = new int[maxChoosableInteger + 1];
        return canWin(state, desiredTotal, new HashMap<String, Boolean>());
    }

    private boolean canWin(int[] state, int total, HashMap<String, Boolean> hmap) {
        String key = Arrays.toString(state);
        if (hmap.containsKey(key)) return hmap.get(key);
        for (int i = 1; i < state.length; i++) {
            if (state[i] == 0) {
                state[i] = 1;
                if(total - i <= 0 || !canWin(state, total - i, hmap)) {
                    hmap.put(key, true);
                    state[i] = 0;
                    return true;
                }
                state[i] = 0;
            }
        }
        hmap.put(key, false);
        return false;
    }

    // leetcode 1235 dynamic programming + binary search
    // dp[t] means the max profit end at t
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        int[][] jobs = new int[n][3];
        for (int i = 0; i < n; ++i) {
            jobs[i] = new int[] {startTime[i], endTime[i], profit[i]};
        }
        Arrays.sort(jobs, (a, b)->a[1] - b[1]); // sort by end times
        TreeMap<Integer, Integer> dp = new TreeMap<>();
        dp.put(0, 0);
        for (int[] job: jobs) {
            int curr = dp.floorEntry(job[0]).getValue() + job[2];
            if (curr > dp.lastEntry().getValue()) {
                dp.put(job[1], curr);
            }
        }
        return dp.lastEntry().getValue();
    }

    public int maximumSum(int[] a) {
        int n = a.length;
        int[] maxEndHere = new int[n], maxStartHere = new int[n];
        maxEndHere[0] = a[0];
        int max = a[0];
        for (int i = 1; i < n; ++i) {
            maxEndHere[i] = Math.max(a[i], maxEndHere[i - 1] + a[i]);
            max = Math.max(max, maxEndHere[i]);
        }
        maxStartHere[n - 1] = a[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            maxStartHere[i] = Math.max(a[i], maxStartHere[i + 1] + a[i]);
        }

        for(int i = 1; i < n - 1; ++i) {
            max = Math.max(max, maxEndHere[i - 1] + maxStartHere[i + 1]);
        }

        return max;
    }

    // leetcode 1143 Longest Common Subsequence
    // If a[i] == b[j], LCS for i and j would be 1 plus LCS till the i-1 and j-1 indexes.
    // Otherwise, we will take the largest LCS if we skip a character from one of the string (max(m[i - 1][j], m[i][j - 1]).
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m+1][n+1];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if(text1.charAt(i) == text2.charAt(j)) {
                    dp[i+1][j+1] = dp[i][j] + 1;
                }
                else {
                    dp[i+1][j+1] = Math.max(dp[i][j+1], dp[i+1][j]);
                }
            }
        }
        return dp[m][n];
    }

    // leetcode 1155 Number of Dice Rolls With Target Sum
    // dynamic programming
    // dp[0][0] = 1, dp[i][t] = sum(dp[i-1][target - j]), where 1<=j<=min(f, target)
    public int numRollsToTarget(int d, int f, int target) {
        double[][] dp = new double[d + 1][target + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= d; ++i) {
            for (int j = 1; j <= f; ++j) {
                for (int k = j; k <= target; ++k) {
                    dp[i][k] = (dp[i][k] + dp[i - 1][k - j]) % (1e9 + 7);
                }
            }
        }
        return (int)dp[d][target];
    }

    // leetcode 1137 N-th Tribonacci Number, iterative solution
    public int tribonacci(int n) {
        if (n == 0) return n;
        int t0 = 0, t1 = 1, t2 = 1, t = 1;
        for (int i = 3; i <=n; ++i) {
            t = t0 + t1 + t2;
            t0 = t1;
            t1 = t2;
            t2 = t;
        }
        return t;
    }

    // leetcode 325 maximum size subarray sum equals k
    public int maxSubArrayLen(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int sum = 0, maxLen = 0;
        for (int i = 0; i < nums.length; ++i) {
            sum += nums[i];
            if (map.containsKey(sum - k)) {
                maxLen = Math.max(maxLen, i - map.get(sum - k));
            }
            map.putIfAbsent(sum, i);
        }
        return maxLen;
    }

    // leetcode 395
    public int longestSubstring(String s, int k) {
        return 0;
    }

    // leetcode 159 longest substring with at most two distinct characters
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        int res = 0, left = 0;
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); ++i) {
            Character c = s.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
            while (map.size() > 2) {
                map.put(s.charAt(left), map.get(s.charAt(left) - 1));
                if (map.get(s.charAt(left)) == 0) map.remove(s.charAt(left));
                ++left;
            }
            res = Math.max(res, i - left + 1);
        }
        return res;
    }

    // leetcode 904 fruit into baskets
    public int totalFruit(int[] tree) {
        int res = 0, left = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < tree.length; ++i) {
            map.put(tree[i], map.getOrDefault(tree[i], 0) + 1);
            while (map.size() > 2) {
                map.put(tree[left], map.get(tree[left]) - 1);
                if (map.get(tree[left]) == 0) map.remove(tree[left]);
                ++left;
            }
            res = Math.max(res, i - left + 1);
        }
        return res;
    }

    // leetcode 42 trapping rain water
    // dynamic programming from left and right
    public int trap(int[] height) {
        return 0;
    }

    // leetcode 3 longest substring without repeating charaters
    // sliding window + hash map
    public int lengthOfLongestSubstring(String s) {
        int left = -1, res = 0, n = s.length();
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            char key = s.charAt(i);
            if (map.containsKey(key) && map.get(key) < i) {
                left = map.get(key);
            }
            map.put(key, i);
            res = Math.max(res, i - left);
        }
        return res;
    }

    // leetcode 56 merge intervals
    public int[][] merge(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        Arrays.sort(intervals, (i1, i2) -> Integer.compare(i1[0], i2[0]));

        List<int[]> result = new ArrayList<>();
        int[] newInterval = intervals[0];
        result.add(newInterval);
        for (int[] interval: intervals) {
            if (interval[0] <= newInterval[1]) {
                newInterval[1] = Math.max(interval[1], newInterval[1]);
            } else {
                newInterval = interval;
                result.add(newInterval);
            }
        }
        return result.toArray(new int[result.size()][]);
    }

    // leetcode 49 group anagrams
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String s: strs) {
            int[] arr = new int[26];
            for (char c: s.toCharArray()) {
                ++arr[c - 'a'];
            }
            String key = Arrays.toString(arr);
            List<String> value = map.getOrDefault(key, new ArrayList<String>());
            value.add(s);
            map.put(key, value);
        }
        return new ArrayList<>(map.values());
    }

    // leetcode 54 spiral matrix
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0) return res;
        int m = matrix.length, n = matrix[0].length;
        int up = 0, down = m - 1, left = 0, right = n - 1;
        while (true) {
            for (int j = left; j <= right; ++j) res.add(matrix[up][j]);
            if (++up > down) break;
            for (int j = up; j <= down; ++j) res.add(matrix[j][right]);
            if (--right < left) break;
            for (int j = right; j >= left; --j) res.add(matrix[down][j]);
            if (--down < up) break;
            for (int j = down; j >= up; --j) res.add(matrix[j][left]);
            if (++left > right) break;
        }
        return res;
    }

    // leetcode 91 decode ways
    // dynamic programming - check if s[i] is 0 or not. basic idea is dp[i] = dp[i-1] + dp[i-2]
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) return 0;
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        for (int i = 1; i < dp.length; ++i) {
            dp[i] = s.charAt(i - 1) == '0' ? 0 : dp[i - 1];
            if (i > 1 && (s.charAt(i - 2) == '1' || (s.charAt(i - 2) == '2' && s.charAt(i - 1) <= '6'))) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[dp.length - 1];
    }

    // leetcode 88 merge sorted array
    // merge from the back
    public void merge(int[] nums1, int m, int[] nums2, int n) {

    }

    // leetcode 332 reconstruct itinerary
    public List<String> findItinerary(List<List<String>> tickets) {
        return null;
    }

    // leetcode 207 course schedule
    // leetcode 210 course schedule II


    // leetcode 380 insert delete getrandom O(1)

    // leetcode 322 coin change
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) return 0;
        if (coins.length == 0 || amount < 0) return -1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);

        for (int i = 1; i <= amount; ++i) {
            for (int j = 0; j < coins.length; ++j) {
                if (i >= coins[j]) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        int res = dp[amount] > amount ? -1 : dp[amount];
        return res;
    }


}




