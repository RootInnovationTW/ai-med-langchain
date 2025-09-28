package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

// HealthCheck 回傳服務狀態
func healthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
	})
}

// Protein Folding 模擬端點
func foldProtein(w http.ResponseWriter, r *http.Request) {
	var input map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	result := map[string]string{
		"sequence": fmt.Sprintf("%v", input["sequence"]),
		"structure": "mock_structure_xyz",
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func main() {
	http.HandleFunc("/health", healthCheck)
	http.HandleFunc("/fold", foldProtein)

	fmt.Println("🚀 Go NIM Service running at http://0.0.0.0:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
